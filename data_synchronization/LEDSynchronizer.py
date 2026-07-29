import numpy as np
import os
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import signal
from scipy.signal import find_peaks
from SyncLogger import SyncLogger, FileAlreadySynchronized
from LED_utils import load_LED_array
from wisci_utils import (
    get_LED_from_nwb,
    get_ecog_freq_from_nwb,
    find_corresponding_nwbs,
    get_duration_osbot,
    get_begin_and_duration_nwb,
    get_begin_and_duration_osbot,
    get_begin_and_duration_mp4,
    fit_video_to_ecog_time,
    format_ecog_sample_as_time,
)
import traceback
from scipy.signal import resample
from tqdm import tqdm
import argparse
import psutil

SYNC_SEARCH_MARGIN_SECONDS = 120  # search +/- this many seconds around the wall-clock-estimated lag, instead of the whole (possibly periodic) recording

mem_bar = None  # dedicated tqdm line for live memory readout, set up in __main__


def log_memory(label=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2  # in MB
    text = f"[MEMORY] {label}: {mem:.1f} MB"
    if mem_bar is not None:
        mem_bar.set_description_str(text)
        mem_bar.refresh()
    else:
        print(text)


def robust_zscore(x, eps=1e-8):
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    scale = 1.4826 * mad  # normal-consistent estimate of std from the MAD
    if scale < eps:
        # MAD collapses to 0 for pulse-like signals that sit at one constant value more than half
        # the time (e.g. the ecog trigger channel while the LED is off) -- fall back to std so the
        # scale doesn't blow up to a tiny near-zero divisor
        scale = np.std(x)
    return (x - median) / (scale + eps)


def plot_with_used_window(
    ax,
    arr,
    x_offset,
    used_start,
    used_end,
    color,
    label,
    pale_alpha=0.15,
    strong_alpha=0.8,
):
    n = len(arr)
    local_start = int(np.clip(used_start - x_offset, 0, n))
    local_end = int(np.clip(used_end - x_offset, 0, n))
    x = np.arange(n) + x_offset

    if local_start > 0:
        ax.plot(x[:local_start], arr[:local_start], color=color, alpha=pale_alpha)
    if local_end < n:
        ax.plot(x[local_end:], arr[local_end:], color=color, alpha=pale_alpha)
    if local_end > local_start:
        ax.plot(
            x[local_start:local_end],
            arr[local_start:local_end],
            color=color,
            alpha=strong_alpha,
            label=label,
        )
    else:
        ax.plot([], [], color=color, alpha=strong_alpha, label=label)


def plot_close_up_window(
    ax,
    sig_a,
    sig_b,
    x_offset,
    color_a,
    color_b,
    label_a,
    label_b,
    around,
    ecog_freq,
    end=False,
):
    """
    Plot `around` samples of sig_a on each side of x_offset (a point in sig_a's index frame),
    with sig_b overlaid on the half where the two signals actually overlap: the right half when
    x_offset is the start of the overlap (end=False), the left half when x_offset is the end of
    the overlap (end=True, sig_b must already be trimmed to the overlapping portion).
    """
    x = np.arange(2 * around)
    ax.plot(
        x, sig_a[x_offset - around : x_offset + around], color=color_a, label=label_a
    )
    if end:
        # NB: sig_b[-around:] would break for around == 0 (Python's -0 == 0, so it would
        # select the *whole* array instead of nothing) -- index from len(sig_b) instead.
        ax.plot(
            np.arange(around),
            sig_b[len(sig_b) - around :],
            color=color_b,
            label=label_b,
            alpha=0.5,
        )
    else:
        ax.plot(
            np.arange(around) + around,
            sig_b[:around],
            color=color_b,
            label=label_b,
            alpha=0.5,
        )
    ax.plot(
        np.arange(int(ecog_freq)),
        np.zeros((int(ecog_freq),)),
        color="black",
        label="1s stretch",
        linewidth=2,
        alpha=0.5,
    )


def plot_extended_window(
    ax,
    sig_a,
    sig_b,
    x_offset,
    color_a,
    color_b,
    label_a,
    label_b,
    length,
    ecog_freq,
    end=False,
):
    """
    Plot `length` samples of the overlap between sig_a and sig_b, starting at x_offset
    (end=False) or ending at x_offset (end=True, sig_b must already be trimmed to the
    overlapping portion).
    """
    x = np.arange(length)
    if end:
        ax.plot(
            x,
            sig_a[x_offset - length : x_offset],
            color=color_a,
            label=label_a,
            alpha=0.5,
        )
        # NB: sig_b[-length:] would break for length == 0, same reasoning as in plot_close_up_window.
        ax.plot(
            x, sig_b[len(sig_b) - length :], color=color_b, label=label_b, alpha=0.5
        )
    else:
        ax.plot(
            x,
            sig_a[x_offset : x_offset + length],
            color=color_a,
            label=label_a,
            alpha=0.5,
        )
        ax.plot(x, sig_b[:length], color=color_b, label=label_b, alpha=0.5)
    ax.plot(
        np.arange(int(ecog_freq)),
        np.zeros((int(ecog_freq),)),
        color="black",
        label="1s stretch",
        linewidth=2,
    )


class LedSynchronizer:
    """
    Synchronizes video and ecog data

    Attributes:
        ecog_server_path (str): Directory path where ecog data is stored.
        output_folder (str): Directory path where synchronized data will be saved.
        log_table_path (str): Path to the CSV log table.
        log_table (pandas.DataFrame): DataFrame loaded from the CSV log table.
        sigma (int): Sigma of the Gaussian filter for smoothing.
        visualize (bool): Flag to enable or disable visualization of the synchronization process.
    """

    def __init__(
        self,
        video_folder,
        led_signals_folder,
        ecog_server_path,
        log_table_path,
        sync_images_folder=None,
        visualize=False,
    ):
        """
        Initializes the Synchronizer with paths and settings for data synchronization.

        Parameters:
            ecog_server_path (str): Path to the directory containing ecog data.
            output_folder (str): Path to the directory where output data will be saved.
            log_table_path (str): Path to the CSV file used as a log table.
            sigma (int): Sigma of the Gaussian filter for smoothing.
            visualize (bool, optional): If True, enables visualization of the process. Defaults to False.
        """
        self.video_folder = video_folder  # path to the video files WITH THE SAVED DURATION IN .NPY FILE!
        self.led_signals_folder = (
            led_signals_folder  # path to the folder with the LED signals
        )
        self.ecog_server_path = ecog_server_path
        self.log_table_path = log_table_path
        self.log = SyncLogger(log_table_path)
        self.visualize = visualize
        self.sync_images_folder = sync_images_folder

        if self.sync_images_folder is not None and not os.path.exists(
            self.sync_images_folder
        ):
            os.makedirs(self.sync_images_folder)

    def save_sync_plots(
        self,
        ecog_signal,
        best_signal,
        best_lag,
        window_start,
        window_end,
        best_corr_array,
        original_video_basename,
        ecog_basename,
        ecog_freq,
        ecog_begin,
    ):
        """
        Save the diagnostic plots for one video/ecog synchronization: close-up and extended
        views at both the start AND the end of the overlap (drift over a long recording can
        make a sync that looks fine at the start drift out of alignment by the end), the whole
        signal with the searched window highlighted, and the correlation curve.
        """
        original_video_basename_noext = os.path.splitext(original_video_basename)[0]

        def out_path(suffix):
            return os.path.join(
                self.sync_images_folder,
                f"{original_video_basename_noext}_{ecog_basename}_{suffix}.png",
            )

        if best_lag < 0:
            plot_lag = -best_lag
            sig_a, sig_b = best_signal, ecog_signal
            label_a, label_b = "From video", "ecog"
            color_a, color_b = "blue", "red"
        else:
            plot_lag = best_lag
            sig_a, sig_b = ecog_signal, best_signal
            label_a, label_b = "ecog", "From video"
            color_a, color_b = "red", "blue"

        # the portion of sig_b that actually overlaps sig_a, and where in sig_a's index frame
        # that overlap begins/ends
        overlap_len = min(len(sig_a) - plot_lag, len(sig_b))
        overlap_end = plot_lag + overlap_len
        sig_b_overlap = sig_b[:overlap_len]

        grid_width = ecog_freq / (1000 / 150)

        def add_time_grid(ax, plot_width, base_offset):
            # fine, fixed-spacing grid (every ~150ms) for the 15s close-up plots, where precise
            # sample-level alignment needs to be visually checkable
            ax.grid(
                True,
                linewidth=0.5,
                linestyle="--",
                color="gray",
                which="both",
                alpha=0.5,
            )
            ax.set_xticks(np.arange(0, plot_width, grid_width))
            new_labels = [
                format_ecog_sample_as_time(label + base_offset, ecog_begin, ecog_freq)
                for label in ax.get_xticks()
            ]
            ax.set_xticklabels(new_labels, fontsize=4, rotation=90)

        def set_wall_time_axis(ax, base_offset):
            # coarser plots (multi-minute / whole-recording) let matplotlib auto-pick tick
            # positions; we just reformat whatever ticks it picks as wall-clock time
            ax.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda x, pos: format_ecog_sample_as_time(
                        x + base_offset, ecog_begin, ecog_freq
                    )
                )
            )
            ax.grid(
                True,
                linewidth=0.5,
                linestyle="--",
                color="gray",
                which="both",
                alpha=0.5,
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)

        # --- Close-up at the start of the overlap -------------------------------------
        around = int(15 * ecog_freq)  # 15 seconds worth of samples at the ecog's rate
        around = min(around, plot_lag, len(sig_a) - plot_lag, len(sig_b))

        fig, ax = plt.subplots(figsize=(15, 5), dpi=200)
        plot_close_up_window(
            ax,
            sig_a,
            sig_b,
            plot_lag,
            color_a,
            color_b,
            label_a,
            label_b,
            around,
            ecog_freq,
            end=False,
        )
        add_time_grid(ax, 2 * around, plot_lag - around)
        ax.legend()
        plt.savefig(out_path("beginning"))
        plt.close(fig)

        # --- Close-up at the end of the overlap -----------------------------------------
        around_end = int(15 * ecog_freq)
        around_end = min(around_end, overlap_end, len(sig_a) - overlap_end, overlap_len)

        fig, ax = plt.subplots(figsize=(15, 5), dpi=200)
        plot_close_up_window(
            ax,
            sig_a,
            sig_b_overlap,
            overlap_end,
            color_a,
            color_b,
            label_a,
            label_b,
            around_end,
            ecog_freq,
            end=True,
        )
        add_time_grid(ax, 2 * around_end, overlap_end - around_end)
        ax.legend()
        plt.savefig(out_path("ending"))
        plt.close(fig)

        # --- First 2 minutes of the overlap ----------------------------------------------
        length = min(int(120 * ecog_freq), len(sig_b), len(sig_a) - plot_lag)

        fig, ax = plt.subplots(figsize=(15, 5), dpi=200)
        plot_extended_window(
            ax,
            sig_a,
            sig_b,
            plot_lag,
            color_a,
            color_b,
            label_a,
            label_b,
            length,
            ecog_freq,
            end=False,
        )
        set_wall_time_axis(ax, plot_lag)
        ax.legend()
        plt.savefig(out_path("two_mins"))
        plt.close(fig)

        # --- Last 2 minutes of the overlap -----------------------------------------------
        length_end = min(int(120 * ecog_freq), overlap_len, overlap_end)

        fig, ax = plt.subplots(figsize=(15, 5), dpi=200)
        plot_extended_window(
            ax,
            sig_a,
            sig_b_overlap,
            overlap_end,
            color_a,
            color_b,
            label_a,
            label_b,
            length_end,
            ecog_freq,
            end=True,
        )
        set_wall_time_axis(ax, overlap_end - length_end)
        ax.legend()
        plt.savefig(out_path("last_two_mins"))
        plt.close(fig)

        # --- Whole signal plot -------------------------------------------------------
        # the ecog trace is drawn pale outside the window that was actually searched,
        # and solid within it, to make the effect of the timestamp-based windowing visible
        fig, ax = plt.subplots(figsize=(15, 5), dpi=200)

        if best_lag < 0:
            video_sig, video_offset, video_color, video_label = sig_a, 0, color_a, label_a
            ecog_sig, ecog_offset, ecog_color, ecog_label = sig_b, plot_lag, color_b, label_b
        else:
            ecog_sig, ecog_offset, ecog_color, ecog_label = sig_a, 0, color_a, label_a
            video_sig, video_offset, video_color, video_label = sig_b, plot_lag, color_b, label_b

        ax.plot(
            np.arange(len(video_sig)) + video_offset,
            video_sig,
            color=video_color,
            label=video_label,
            alpha=0.5,
        )
        plot_with_used_window(
            ax,
            ecog_sig,
            ecog_offset,
            window_start,
            window_end,
            color=ecog_color,
            label=ecog_label,
        )
        ax.plot(
            np.arange(int(ecog_freq)),
            np.zeros((int(ecog_freq),)),
            color="black",
            label="1s stretch",
            linewidth=2,
        )
        set_wall_time_axis(ax, 0)

        ax.legend()
        plt.savefig(out_path("whole"))
        plt.close(fig)

        # --- Correlation plot ----------------------------------------------------------
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(window_start, window_start + len(best_corr_array)),
            best_corr_array,
        )
        set_wall_time_axis(ax, 0)
        plt.savefig(out_path("corr"))
        plt.close(fig)

    def sync_and_optimize_freq(
        self,
        ecog_signal,
        video_signal,
        video_duration,
        original_video_basename=None,
        ecog_basename=None,
        expected_lag_seconds=None,
        *,
        ecog_freq,
        ecog_begin=None,
    ):
        # trying different frequencies as ecog and realsense are both imprecise in their sampling
        # frequencies so I adjust it like this - we find the freq that gets the best result :)
        video_freq = len(video_signal) / video_duration

        # upsample the video to the ecog frequency
        new_num_samples = int(len(video_signal) * (ecog_freq / video_freq))

        # restrict the search to a window around the wall-clock-estimated lag, instead of the
        # whole (possibly hours-long, periodically-blinking) ecog recording -- avoids picking up
        # a spurious correlation peak from a repeat of the same LED flash elsewhere in the recording
        margin_samples = int(SYNC_SEARCH_MARGIN_SECONDS * ecog_freq)
        max_resampled_len = int(new_num_samples + 2000)  # upper bound of the resample search below
        if expected_lag_seconds is not None:
            expected_lag_samples = int(round(expected_lag_seconds * ecog_freq))
            window_start = max(0, expected_lag_samples - margin_samples)
            window_end = min(len(ecog_signal), expected_lag_samples + margin_samples + max_resampled_len)
            if window_end - window_start < max_resampled_len:
                tqdm.write(f"Expected lag {expected_lag_seconds:.1f}s (+/- {SYNC_SEARCH_MARGIN_SECONDS}s) falls outside the ecog recording -- falling back to full-signal search.")
                window_start, window_end = 0, len(ecog_signal)
        else:
            window_start, window_end = 0, len(ecog_signal)
        ecog_search_window = ecog_signal[window_start:window_end]

        best_corr = -1
        best_lag = 0
        best_n_samples = 0
        best_signal = None
        best_corr_array = None

        search_desc = f"{original_video_basename or '?'} vs {ecog_basename or '?'}"
        for n_samples in tqdm(np.linspace(new_num_samples - 2000, new_num_samples + 2000, 250), desc=search_desc, position=2, leave=False):
            log_memory(f"before resample {n_samples}")
            resampled_video = resample(video_signal, int(n_samples))
            log_memory(f"after resample, before correlate {n_samples}")
            corr, lag, corr_array = self.synchronize_by_LED(ecog_search_window, resampled_video)
            lag = lag + window_start  # convert back to an index into the full ecog_signal
            log_memory("after correlate")
            if corr > best_corr:
                best_corr, best_lag = corr, lag
                best_signal = resampled_video
                best_n_samples = n_samples
                best_corr_array = corr_array

        if self.sync_images_folder is not None and original_video_basename is not None:
            self.save_sync_plots(
                ecog_signal,
                best_signal,
                best_lag,
                window_start,
                window_end,
                best_corr_array,
                original_video_basename,
                ecog_basename,
                ecog_freq,
                ecog_begin,
            )

        peaks, _ = find_peaks(
            best_corr_array,
            height=0.5 * np.max(best_corr_array),
            distance=int(ecog_freq) * 2,
        )  # TODO: fix distance by the freq
        second_largest_corr_peak = 0
        if len(peaks) < 2:
            second_largest_corr_peak = 0
        else:
            peak_values = np.abs(best_corr_array)[peaks]
            sorted_unique_peak_values = np.sort(np.unique(peak_values))
            second_largest_corr_peak = sorted_unique_peak_values[-2]

        return best_corr, best_lag, best_n_samples, len(peaks), second_largest_corr_peak

    def sync_with_led(
        self,
        video_fullpath,
        led_signal_full_path,
        ecog_file,
        log=True,
        output_path_manual=None,
    ):
        # checked first, before any (expensive) ecog reading: a marker file means labeling already
        # established this video has no usable LED. sync_video_folder normally catches this before
        # calling us at all -- this guards direct calls to sync_with_led.
        video_led, led_error = load_LED_array(led_signal_full_path)
        if led_error is not None:
            raise RuntimeError(led_error)

        ecog_signal, _ = get_LED_from_nwb(ecog_file)
        ecog_freq = get_ecog_freq_from_nwb(ecog_file)
        try:
            ecog_begin, ecog_duration = get_begin_and_duration_nwb(ecog_file)
        except Exception:
            ecog_begin, ecog_duration = None, None
            tqdm.write(
                f"Could not read session_start_time from {ecog_file}, will fall back to full-signal search."
            )

        original_num_frames_video = len(video_led)  # frame count before any resampling -- needed downstream to map ECoG samples <-> video frames
        video_basename = os.path.basename(video_fullpath)
        ecog_basename = os.path.basename(str(Path(ecog_file).parent))
        video_fullpath_lower = video_fullpath.lower()
        if video_fullpath_lower.endswith(".mkv"):
            video_duration = get_duration_osbot(video_fullpath)
            tqdm.write(f"Video duration from .mkv timestamps: {video_duration:.3f}s (default-duration diff: {video_duration - (0.02 * (len(video_led) - 1) + 1):.3f}s)")
            try:
                video_begin, _ = get_begin_and_duration_osbot(video_fullpath)
            except Exception:
                video_begin = None
        elif video_fullpath_lower.endswith(".mp4"):
            try:
                video_begin, video_duration = get_begin_and_duration_mp4(video_fullpath)
                tqdm.write(f"Video duration from .XML metadata: {video_duration:.3f}s")
            except Exception:
                video_duration = (
                    len(video_led) / 30.0
                )  # assume fs of 30 if cannot read fs... maybe need something different
                video_begin = None
                tqdm.write(f"Could not read video timing metadata for {video_basename}, assuming 30 fps.")

        if ecog_begin is not None and video_begin is not None:
            expected_lag_seconds = video_begin - ecog_begin
            tqdm.write(f"Approximate lag from wall-clock timestamps: {expected_lag_seconds:.1f}s -> restricting sync search to +/-{SYNC_SEARCH_MARGIN_SECONDS}s around it")
        else:
            expected_lag_seconds = None
            tqdm.write("No reliable wall-clock timestamps for this video/ecog pair, falling back to full-signal search.")

        # median/MAD z-score: keeps the "LED off" baseline near 0 for both signals. The previous
        # min/max rescale left a large non-zero DC term (both signals sit near one extreme of
        # [-0.5, 0.5] while the LED is off), which dominates the cross-correlation sum at every lag
        # and makes the find_peaks height threshold in sync_and_optimize_freq meaningless. Median/MAD
        # are also robust to the few dim frames at the start/end of a video, so the manual edge-trim
        # previously needed there is no longer necessary.
        video_led = robust_zscore(video_led)
        ecog_signal = robust_zscore(ecog_signal)

        if video_duration > 15:  # process videos that are at least 15s long
            (
                best_corr,
                best_lag,
                best_n_samples,
                best_total_peaks,
                best_second_largest_corr_peak,
            ) = self.sync_and_optimize_freq(
                ecog_signal,
                video_led,
                video_duration,
                video_basename,
                ecog_basename,
                expected_lag_seconds=expected_lag_seconds,
                ecog_freq=ecog_freq,
                ecog_begin=ecog_begin,
            )
            tqdm.write(
                f"  -> corr={best_corr:.1f}  lag={best_lag} ({best_lag / ecog_freq:.2f}s)"
            )
            sync_failed = 0

        else:
            best_corr, best_lag, best_n_samples, best_total_peaks, best_second_largest_corr_peak = -1, -1, -1, -1, -1

            sync_failed = 1
            self.log.update_log(video_basename, 'video_duration', video_duration)
            self.log.update_log(video_basename, 'sync_error_msg', "Video too short (under 15s).")

        slope, intercept = fit_video_to_ecog_time(
            ecog_begin=ecog_begin,
            video_begin=video_begin,
            video_duration=video_duration,
            ecog_freq=ecog_freq,
            lag=best_lag,
            best_n_samples=best_n_samples,
            sync_failed=sync_failed,
        )

        if log:
            self.log.update_log(video_basename, "video_duration", video_duration)
            self.log.update_log(video_basename, "video_timestamp", video_begin)
            self.log.update_log(video_basename, "ecog_timestamp", ecog_begin)
            self.log.update_log(video_basename, "ecog_duration", ecog_duration)
            self.log.update_log(video_basename, "ecog_frequency", ecog_freq)
            self.log.update_log(
                video_basename, "original_num_frames_video", original_num_frames_video
            )
            self.log.update_log(
                video_basename, "best_resampled_video_num_frames", int(best_n_samples)
            )
            self.log.update_log(video_basename, "path_ecog", ecog_file)
            self.log.update_log(video_basename, "corr", best_corr)
            self.log.update_log(video_basename, "lag", best_lag)
            self.log.update_log(
                video_basename, "expected_lag_seconds", expected_lag_seconds
            )
            self.log.update_log(video_basename, "sync_failed", sync_failed)
            self.log.update_log(
                video_basename,
                "additional_peaks_per_million",
                (best_total_peaks - 1) / len(ecog_signal) * 1000000,
            )
            self.log.update_log(
                video_basename,
                "best_second_largest_corr_peak",
                best_second_largest_corr_peak,
            )
            self.log.update_log(video_basename, "slope", slope)
            self.log.update_log(video_basename, "intercept", intercept)
        else:
            # write the logs into a file logs_manual.txt in the output_path_manual (but if the file exists, append to it)
            with open(os.path.join(output_path_manual, "logs_manual.txt"), "a") as f:
                f.write(f"Path to the best ecog file: {ecog_file}\n")
                f.write(f"Normalized correlation: {best_corr}\n")
                f.write(f"Lag: {best_lag}\n")
                f.write(f"Synchronization failed: 0\n")

    def sync_video_folder(self, extension):

        extension = extension.lower()
        sessions_folder = os.listdir(self.video_folder)

        # collect every matching video across all sessions up-front so we can report overall
        # progress ("Processing 24/120") instead of restarting the count at every session
        video_paths = []
        for session in sessions_folder:
            session_path = os.path.join(self.video_folder, session)
            video_paths.extend(
                str(file) for file in Path(session_path).rglob("*")
                if file.is_file() and extension in file.name.lower()
            )
        total_videos = len(video_paths)

        for i, video_path in tqdm(list(enumerate(video_paths, start=1)), desc="Processing videos", unit="video", position=0, leave=True):
            video_basename = os.path.basename(video_path)
            tqdm.write(f"\n{'=' * 90}\nProcessing {i}/{total_videos}: {video_basename}\n{'=' * 90}")
            tqdm.write(video_path)
            log_memory(f"start of file {video_path}")
            try:
                self.log.process_new_file(video_basename)
            except FileAlreadySynchronized as e:
                tqdm.write(f"{e}\nSkipping {video_basename}")
                continue

            video_basename_noext = os.path.splitext(video_basename)[0]
            led_signal_full_path = os.path.join(self.led_signals_folder, video_basename_noext + "_LED_signal.npy")

            # a marker file means labeling already established this video has no usable LED --
            # record the reason and move on without searching for (or opening) any ecog file
            if os.path.exists(led_signal_full_path):
                _, led_error = load_LED_array(led_signal_full_path)
                if led_error is not None:
                    tqdm.write(f"  -> {led_error}. Skipping synchronization.")
                    self.log.update_log(video_basename, "sync_error_msg", led_error)
                    self.log.update_log(video_basename, "sync_failed", 1)
                    self.log.save_to_csv()  # the ecog loop's finally: below is what normally saves
                    continue

            try:
                ecog_files = find_corresponding_nwbs(video_path, self.video_folder, self.ecog_server_path)
            except Exception:
                tqdm.write(traceback.format_exc())
                self.log.update_log(video_basename, "sync_error_msg", traceback.format_exc())
                self.log.update_log(video_basename, "sync_failed", 1)
                ecog_files = []

            if len(ecog_files) == 0:
                self.log.update_log(video_basename, "no ecog files", 1)
            for ecog_file in ecog_files:
                ecog_basename = os.path.basename(str(Path(ecog_file).parent))
                tqdm.write(f"  -- syncing against ecog: {ecog_basename} --")
                log_memory(f"start of ecog {ecog_file}")
                try:
                    self.sync_with_led(
                        video_path,
                        led_signal_full_path,
                        ecog_file,
                    )
                except Exception:
                    tqdm.write(traceback.format_exc())
                    self.log.update_log(video_basename, "sync_error_msg", traceback.format_exc())
                    self.log.update_log(video_basename, "sync_failed", 1)
                finally:
                    self.log.save_to_csv()

    def synchronize_by_LED(self, ecog_signal, video, visualize=False):
        normalized_sig1 = ecog_signal.reshape(-1,)
        normalized_sig2 = video

        correlation = signal.correlate(normalized_sig1, normalized_sig2, mode="valid")
        lags = signal.correlation_lags(normalized_sig1.size, normalized_sig2.size, mode="valid")
        lag = lags[np.argmax(np.abs(correlation))]
        if visualize:
            print(lag)
            plt.plot(np.abs(correlation))
            plt.show()

            plt.plot(
                np.abs(
                    correlation[
                        np.argmax(np.abs(correlation)) - 600 : np.argmax(
                            np.abs(correlation)
                        )
                        + 600
                    ]
                )
            )
            plt.title("Peak close-up")
            plt.show()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(len(normalized_sig1)), y=normalized_sig1, mode='lines', name='ECoG', line=dict(color='red')))

            # we computed abs(correlation), to visualize the signals, let's align them
            # (if they were anticorrelated, just put minus before the signal)
            if correlation[np.argmax(np.abs(correlation))] > 0:
                print("positive")
                fig.add_trace(go.Scatter(x=np.arange(len(normalized_sig2)) + lag, y=normalized_sig2, mode='lines', name='From video', line=dict(color='blue')))
            else:
                print("negative")
                fig.add_trace(go.Scatter(x=np.arange(len(normalized_sig2)) + lag, y=-normalized_sig2, mode='lines', name='From video', line=dict(color='blue')))
            fig.show()

        return np.max(np.abs(correlation)), lag, np.abs(correlation)


if __name__ == "__main__":
    visualize = False

    parser = argparse.ArgumentParser(description="Synchronize ECoG and Video data using the LED signals.")
    parser.add_argument("video_folder", help="Path to the folder containing video files.")
    parser.add_argument("led_signals_folder", help="Path to the folder containing led position .npy files.")
    parser.add_argument("ecog_folder", help="Path to the folder with ecog .nwb files.")
    parser.add_argument("log_table_path", help="Where to log the synchronization.")
    parser.add_argument("sync_images_path", help="Where to store the logs of the synchronization.")
    parser.add_argument(
        "extension",
        type=lambda v: (v if v.startswith(".") else f".{v}").lower(),
        choices=[".mp4", ".mkv"],
        help="video extension (.mp4 or .mkv, case-insensitive)",
    )
    args = parser.parse_args()

    mem_bar = tqdm(total=0, position=1, bar_format="{desc}", leave=True)

    synchronizer = LedSynchronizer(
        args.video_folder,
        args.led_signals_folder,
        args.ecog_folder,
        args.log_table_path,
        args.sync_images_path,
        visualize,
    )
    synchronizer.sync_video_folder(args.extension)

    mem_bar.close()

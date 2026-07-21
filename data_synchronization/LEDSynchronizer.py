import numpy as np
import os
from pathlib import Path
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from SyncLogger import SyncLogger, FileAlreadySynchronized
from wisci_utils import (
    get_LED_from_nwb,
    find_corresponding_nwbs,
    get_duration_mp4,
    get_duration_osbot,
)
import traceback
from scipy.signal import resample
from tqdm import tqdm
import argparse
import psutil

ECOG_FREQ = 585.9375  # Hz, the sampling frequency of the ecog device
PLOT_AROUND_SAMPLES = int(15 * ECOG_FREQ) # number of samples to plot around the lag point (15 seconds worth of samples at 585 Hz)

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
    def __init__(self, video_folder, led_signals_folder, ecog_server_path, log_table_path, sync_images_folder=None, visualize=False):
        """
        Initializes the Synchronizer with paths and settings for data synchronization.

        Parameters:
            ecog_server_path (str): Path to the directory containing ecog data.
            output_folder (str): Path to the directory where output data will be saved.
            log_table_path (str): Path to the CSV file used as a log table.
            sigma (int): Sigma of the Gaussian filter for smoothing.
            visualize (bool, optional): If True, enables visualization of the process. Defaults to False.
        """
        self.video_folder = video_folder # path to the video files WITH THE SAVED DURATION IN .NPY FILE!
        self.led_signals_folder = led_signals_folder # path to the folder with the LED signals
        self.ecog_server_path = ecog_server_path
        self.log_table_path = log_table_path
        self.log = SyncLogger(log_table_path)
        self.visualize = visualize
        self.sync_images_folder = sync_images_folder

        if self.sync_images_folder is not None and not os.path.exists(self.sync_images_folder):
            os.makedirs(self.sync_images_folder)

    def sync_and_optimize_freq(self, ecog_signal, forehead, accelerometer_duration, video_duration, original_video_basename=None, ecog_basename=None):
        # trying different frequencies as ecog and realsense are both imprecise in their sampling
        # frequencies so I adjust it like this - we find the freq that gets the best result :)
        # print("SYNCING. ACCELEROMETR DATA LENGTH:", len(ecog_signal))
        # print("FOREHEAD POINTS LENGTH:", len(forehead_points))

        # ecog_freq = len(ecog_signal) / accelerometer_duration
        ecog_freq = ECOG_FREQ
        video_freq = len(forehead) / video_duration

        # upsample the video to the ecog frequency
        new_num_samples = int(len(forehead) * (ecog_freq / video_freq))

        best_corr = -1
        best_lag = 0
        best_n_samples = 0
        corrs = []

        best_signal = None

        for n_samples in tqdm(np.linspace(new_num_samples - 2000, new_num_samples + 2000, 250)):
            log_memory(f"before resample {n_samples}")
            resampled_video = resample(forehead, int(n_samples))
            # plot both signals
            log_memory(f"before resample {n_samples}")
            corr, lag, corr_array = self.synchronize_by_LED(ecog_signal, resampled_video)
            log_memory("after correlate")
            corrs.append(corr)
            if corr > best_corr:
                best_corr, best_lag = corr, lag
                best_signal = resampled_video
                best_n_samples = n_samples
                best_corr_array = corr_array
    
        resampled_video = resample(forehead, int(best_n_samples))
        if self.sync_images_folder is not None and original_video_basename is not None:

            original_video_basename_noext = os.path.splitext(original_video_basename)[0]

            # --- Prepare plotting-only variables -----------------------------------------
            if best_lag < 0:
                plot_lag = -best_lag
                sig_a = best_signal
                sig_b = ecog_signal
                label_a, label_b = "From video", "ecog"
                color_a, color_b = "blue", "red"
            else:
                plot_lag = best_lag
                sig_a = ecog_signal
                sig_b = best_signal
                label_a, label_b = "ecog", "From video"
                color_a, color_b = "red", "blue"

            # --- Around window plot ------------------------------------------------------
            around = PLOT_AROUND_SAMPLES
            around = min(around, plot_lag)
            around = min(around, len(sig_a) - plot_lag, len(sig_b))

            fig, ax = plt.subplots(figsize=(15, 5), dpi=200)

            x = np.arange(2 * around)
            y = sig_a[plot_lag - around : plot_lag + around]

            ax.plot(x, y, color=color_a, label=label_a)

            ax.plot(np.arange(around) + around, sig_b[:around], color=color_b, label=label_b, alpha=0.5)

            ax.plot(np.arange(int(ECOG_FREQ)), np.zeros((int(ECOG_FREQ),)), color="black", label="1s stretch", linewidth=2, alpha=0.5)

            plot_width = 2 * around
            grid_width = 585 / (1000 / 150)

            # Add grid lines with predefined width
            ax.grid(True, linewidth=0.5, linestyle='--', color='gray', which='both', alpha=0.5)

            # Set x ticks with predefined width
            ax.set_xticks(np.arange(0, plot_width, grid_width))
            
            ax.legend()
            new_labels = [str(int(label) + plot_lag - around) for label in ax.get_xticks()]
            ax.set_xticklabels(new_labels, fontsize=4, rotation=90)

            plt.savefig(os.path.join(self.sync_images_folder, original_video_basename_noext + "_" + ecog_basename + "_beginning.png",),)
            plt.close(fig)

            # --- First 2 minutes plot ----------------------------------------------------
            length = int(120 * ECOG_FREQ)  # 2 minutes worth of samples
            length = min(length, len(sig_b[:length]), len(sig_a[plot_lag : plot_lag + length]))

            _, ax = plt.subplots(figsize=(15, 5), dpi=200)
            ax.plot(np.arange(length), sig_a[plot_lag : plot_lag + length], color=color_a, label=label_a, alpha=0.5)
            ax.plot(np.arange(length), sig_b[:length], color=color_b, label=label_b, alpha=0.5)
            ax.plot(np.arange(int(ECOG_FREQ)), np.zeros((int(ECOG_FREQ),)), color="black", label="1s stretch", linewidth=2)

            ax.legend()

            plt.savefig(os.path.join(self.sync_images_folder, original_video_basename_noext + "_" + ecog_basename + "_two_mins.png",))
            plt.close(fig)

            # --- Whole signal plot -------------------------------------------------------
            _, ax = plt.subplots(figsize=(15, 5), dpi=200)

            ax.plot(np.arange(len(sig_a)), sig_a, color=color_a, label=label_a, alpha=0.5)
            ax.plot(np.arange(len(sig_b)) + plot_lag, sig_b, color=color_b, label=label_b, alpha=0.5)
            ax.plot(np.arange(int(ECOG_FREQ)), np.zeros((int(ECOG_FREQ),)), color="black", label="1s stretch", linewidth=2)

            ax.legend()

            plt.savefig(os.path.join(self.sync_images_folder, original_video_basename_noext + "_" + ecog_basename + "_whole.png",))
            plt.close(fig)

            plt.figure()
            plt.plot(best_corr_array)

            plt.savefig(os.path.join(self.sync_images_folder, original_video_basename_noext + "_" + ecog_basename + "_corr.png",))
            plt.close(fig)
        peaks, _ = find_peaks(best_corr_array, height=0.5 * np.max(best_corr_array), distance=int(ECOG_FREQ) * 2)  # TODO: fix distance by the freq
        second_largest_corr_peak = 0
        if len(peaks) < 2:
            second_largest_corr_peak = 0
        else:
            peak_values = np.abs(best_corr_array)[peaks]
            sorted_unique_peak_values = np.sort(np.unique(peak_values))
            second_largest_corr_peak = sorted_unique_peak_values[-2]

        return best_corr, best_lag, best_n_samples, len(peaks), second_largest_corr_peak

    def sync_with_led(self, video_fullpath, led_signal_full_path, ecog_file, log=True, output_path_manual=None):
        ecog_signal, ecog_duration = get_LED_from_nwb(ecog_file)

        video_led = np.load(led_signal_full_path)
        video_basename = os.path.basename(video_fullpath)
        ecog_basename = os.path.basename(str(Path(ecog_file).parent))
        # video_duration = get_duration_osbot(video_fullpath)
        video_fullpath_lower = video_fullpath.lower()
        if video_fullpath_lower.endswith(".mkv"):
            video_duration = get_duration_osbot(video_fullpath)
            print(f"Video duration from .mkv timestamps: {video_duration} seconds, difference from the default duration: {video_duration - (0.02 * (len(video_led) - 1) + 1)} seconds")
            # video_duration = 0.02 * (len(video_led) - 1) + 1
        elif video_fullpath_lower.endswith(".mp4"):
            try:
                video_duration = get_duration_mp4(video_fullpath)
                print(video_duration)
            except:
                video_duration = (
                    len(video_led) / 30.0
                )  # assume fs of 30 if cannot read fs... maybe need something different
                print("default fs")

        # video_led = video_led - np.min(video_led[1000:]) #np.mean(sig1) # TODO: here was min(), but I changed it because at the beginning the lighting is very low
        # ecog_signal = ecog_signal - np.min(ecog_signal[1000:]) # TODO: here was min(), but I changed it because at the beginning the lighting is very low
        # video_led = video_led / np.max(video_led) - 0.5
        # ecog_signal = ecog_signal / np.max(ecog_signal) - 0.5

        video_led = video_led - np.max(video_led) #np.mean(sig1) # TODO: here was min(), but I changed it because at the beginning the lighting is very low
        ecog_signal = ecog_signal - np.min(ecog_signal) # TODO: here was min(), but I changed it because at the beginning the lighting is very low
        video_led = video_led / (np.max(video_led) - min(video_led[90:-90])) + 0.5
        ecog_signal = ecog_signal / np.max(ecog_signal) - 0.5

        if video_duration > 15: # process videos that are at least 15s long
            best_corr, best_lag, best_n_samples, best_total_peaks, best_second_largest_corr_peak = self.sync_and_optimize_freq(ecog_signal, video_led, ecog_duration, video_duration, video_basename, ecog_basename)
            print(f"Going to update log")
            print(f"File: {video_basename}")
            print(f"\tBest correlation: {best_corr}\n\tLag: {best_lag}")
            sync_failed = 0

        else:
            best_corr, best_lag, best_n_samples, best_total_peaks, best_second_largest_corr_peak = -1, -1, -1, -1, -1

            sync_failed = 1
            self.log.update_log(video_basename, 'video_duration', video_duration)
            self.log.update_log(video_basename, 'sync_error_msg', "Video too short (under 15s).")


        if log:
            self.log.update_log(video_basename, 'video_duration', video_duration)
            self.log.update_log(video_basename, 'frames', best_n_samples)
            self.log.update_log(video_basename, 'path_ecog', ecog_file)
            self.log.update_log(video_basename, 'corr', best_corr)
            self.log.update_log(video_basename, 'lag', best_lag)
            self.log.update_log(video_basename, 'sync_failed', sync_failed)
            self.log.update_log(video_basename, 'additional_peaks_per_million', (best_total_peaks-1)/len(ecog_signal) * 1000000)
            self.log.update_log(video_basename, 'best_second_largest_corr_peak', best_second_largest_corr_peak)
        else:
            # write the logs into a file logs_manual.txt in the output_path_manual (but if the file exists, append to it)
            with open(os.path.join(output_path_manual, 'logs_manual.txt'), 'a') as f:
                f.write(f"Path to the best ecog file: {ecog_file}\n")
                f.write(f"Normalized correlation: {best_corr}\n")
                f.write(f"Lag: {best_lag}\n")
                f.write(f"Synchronization failed: 0\n")

    def sync_video_folder(self, extension):

        extension = extension.lower()
        sessions_folder = os.listdir(self.video_folder)

        for session in sessions_folder:
            session_path = os.path.join(self.video_folder, session)
            video_files = [str(file) for file in Path(session_path).rglob("*") if file.is_file() and extension in file.name.lower()]

            for video_path in video_files:
                print(video_path)
                log_memory(f"start of file {video_path}")
                video_basename = os.path.basename(video_path)
                try:
                    self.log.process_new_file(video_basename)
                except FileAlreadySynchronized as e:
                    print(e)
                    print(f"Skipping {video_basename}")
                    continue

                print("Processing: ", video_basename)
                video_basename_noext = os.path.splitext(video_basename)[0]
                try:
                    ecog_files = find_corresponding_nwbs(video_path, self.video_folder, self.ecog_server_path)
                except Exception as e:
                    print(traceback.format_exc())
                    self.log.update_log(video_basename, "sync_error_msg", traceback.format_exc())
                    self.log.update_log(video_basename, "sync_failed", 1)
                    ecog_files = []

                if len(ecog_files) == 0:
                    self.log.update_log(video_basename, "no ecog files", 1)
                for ecog_file in ecog_files:
                    log_memory(f"start of ecog {ecog_file}")
                    try:
                        self.sync_with_led(
                            video_path,
                            os.path.join(self.led_signals_folder, video_basename_noext + "_LED_signal.npy"),
                            ecog_file,
                        )
                    except Exception as e:
                        print(traceback.format_exc())
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

            plt.plot(np.abs(correlation[np.argmax(np.abs(correlation))-600:np.argmax(np.abs(correlation))+600]))
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
    

if __name__=="__main__":

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

    synchronizer = LedSynchronizer(args.video_folder, args.led_signals_folder, args.ecog_folder, args.log_table_path, args.sync_images_path, visualize)
    synchronizer.sync_video_folder(args.extension)

    mem_bar.close()
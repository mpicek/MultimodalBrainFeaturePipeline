import pandas as pd
import numpy as np
import re
from datetime import datetime
import xml.etree.ElementTree as ET
from pynwb import NWBHDF5IO
from pathlib import Path


def get_LED_from_nwb(nwb_path):
    try:
        with NWBHDF5IO(nwb_path, "r") as io:
            nwb = io.read()
            trigger = nwb.processing["behavior"].data_interfaces["Trigger"].data[:]
            print(trigger.shape)

            trigger = np.nan_to_num(trigger)
            duration = (
                trigger.shape[0]
                / nwb.processing["behavior"].data_interfaces["Trigger"].rate
            )

            return trigger, duration
    except:  # noqa: E722
        try:
            with NWBHDF5IO(nwb_path, "r") as io:
                nwb = io.read()
                trigger = (
                    nwb.processing["behavior"].data_interfaces["trigger_ch"].data[:]
                )
                trigger = np.nan_to_num(trigger)
                duration = (
                    trigger.shape[0]
                    / nwb.processing["behavior"].data_interfaces["trigger_ch"].rate
                )
                return trigger, duration
        except:
            print(f"No trigger? check file {nwb_path}")
            return


def get_ecog_freq_from_nwb(nwb_path):
    """Read the ecog device's actual sampling rate (Hz) from the nwb Trigger channel."""
    try:
        with NWBHDF5IO(nwb_path, "r") as io:
            nwb = io.read()
            return nwb.processing["behavior"].data_interfaces["Trigger"].rate
    except:  # noqa: E722
        with NWBHDF5IO(nwb_path, "r") as io:
            nwb = io.read()
            return nwb.processing["behavior"].data_interfaces["trigger_ch"].rate


def get_duration_osbot(video_path):
    timestamps_path = re.sub(
        r"\.mkv$", "_timestamps.csv", video_path, flags=re.IGNORECASE
    )
    final_time = pd.read_csv(timestamps_path)["pts_time"].values[-1]
    return final_time


def get_begin_and_duration_osbot(file_path):
    df_path = re.sub(r"\.mkv$", "_timestamps.csv", file_path, flags=re.IGNORECASE)
    df_timestamps = pd.read_csv(df_path)
    begin = datetime.fromtimestamp(df_timestamps["demux_timestamp"][0])
    end = datetime.fromtimestamp(
        df_timestamps["demux_timestamp"][len(df_timestamps) - 1]
    )

    duration = end - begin

    return begin.timestamp(), duration.total_seconds()


def get_begin_and_duration_mp4(file_path):
    tree = ET.parse(re.sub(r"\.mp4$", "M01.XML", file_path, flags=re.IGNORECASE))
    root = tree.getroot()

    # --- Extract values ---
    frames = int(root.find(".//{*}Duration").attrib["value"])
    fps = int(root.find(".//{*}LtcChangeTable").attrib["tcFps"])

    creation_str = root.find(".//{*}CreationDate").attrib["value"]

    # --- Parse creation date ---
    dt = datetime.fromisoformat(creation_str)
    begin = dt.timestamp()

    duration_seconds = frames / fps

    return begin, duration_seconds


def get_begin_and_duration_nwb(file_path):

    with NWBHDF5IO(file_path, "r") as io:
        nwbfile = io.read()
        begin = nwbfile.session_start_time
        duration = (
            nwbfile.acquisition["RawEcoG"].data.shape[0]
            / nwbfile.acquisition["RawEcoG"].rate
        )
        """duration = (
            nwbfile.processing["behavior"].data_interfaces["Trigger"].data.shape[0]
            / nwbfile.processing["behavior"].data_interfaces["Trigger"].rate
        )"""

    return begin.timestamp(), duration


def get_duration_mp4(path_video):

    tree = ET.parse(re.sub(r"\.mp4$", "M01.XML", path_video, flags=re.IGNORECASE))
    root = tree.getroot()

    # --- Extract values ---
    frames = int(root.find(".//{*}Duration").attrib["value"])
    fps = int(root.find(".//{*}LtcChangeTable").attrib["tcFps"])
    duration_seconds = frames / fps

    return duration_seconds


def format_ecog_sample_as_time(sample_index, ecog_begin, ecog_freq):
    """
    Convert an (absolute) ecog sample index to a wall-clock time string, using the ecog's own
    timestamp+rate (ecog_begin + sample_index/ecog_freq) -- exact, no fitting needed, since both
    signals in these diagnostic plots are already expressed in the same ecog-sample-index frame
    (the video signal via `lag`). Falls back to elapsed seconds if ecog_begin is unavailable
    (session_start_time couldn't be read from the nwb file).
    """
    seconds = sample_index / ecog_freq
    if ecog_begin is not None:
        return datetime.fromtimestamp(ecog_begin + seconds).strftime("%H:%M:%S.%f")[:-3]
    return f"{seconds:.2f}s"


def fit_video_to_ecog_time(
    *,
    ecog_begin,
    video_begin,
    video_duration,
    ecog_freq,
    lag,
    best_n_samples,
    sync_failed,
):
    """
    Fit the linear relationship between the ECoG's and the video's own wall clocks (both Unix
    epoch seconds): ECoG_time = slope * video_time + intercept.

    slope should be close to 1 (deviates only by the devices' relative clock-rate error). intercept
    is NOT expected to be close to 0 -- since video_time/ECoG_time are raw Unix-epoch seconds
    (~1.7e9), even a tiny (e.g. 1e-5) deviation of slope from 1 gets scaled by video_begin into an
    intercept of thousands of seconds. That's expected: intercept only makes sense combined with
    slope in the full formula (slope * t + intercept reproduces the correct ECoG_time), not as a
    standalone "small correction" value.

    Returns (None, None) if the fit isn't available (missing wall-clock timestamps, or a failed
    sync where lag/best_n_samples are -1 sentinels).
    """
    if sync_failed or ecog_begin is None or video_begin is None:
        return None, None
    slope = best_n_samples / (video_duration * ecog_freq)
    intercept = ecog_begin + lag / ecog_freq - slope * video_begin
    return slope, intercept


def find_corresponding_nwbs(mp4_total_path, mp4_root_path, ecog_path):

    relative = Path(mp4_total_path).relative_to(mp4_root_path)
    first_folder = relative.parts[0]  # name of the session folder

    ecog_to_check_path = Path(ecog_path) / first_folder

    files_nwb = [str(file) for file in Path(ecog_to_check_path).rglob("*.nwb*")]

    df = pd.DataFrame(columns=["path", "type", "begin", "duration"])

    for nwb in files_nwb:
        begin, duration = get_begin_and_duration_nwb(nwb)
        toadd = pd.DataFrame(
            [{"path": nwb, "type": "nwb", "begin": begin, "duration": duration}]
        )
        df = pd.concat([df, toadd])
    # add mp4 infos
    mp4_total_path_lower = mp4_total_path.lower()
    if mp4_total_path_lower.endswith("mkv"):
        begin, duration = get_begin_and_duration_osbot(mp4_total_path)
    elif mp4_total_path_lower.endswith(".mp4"):
        begin, duration = get_begin_and_duration_mp4(mp4_total_path)
    toadd = pd.DataFrame(
        [
            {
                "path": mp4_total_path,
                "type": "video",
                "begin": begin,
                "duration": duration,
            }
        ]
    )
    df = pd.concat([df, toadd])
    overlaps = find_overlaps(df, "video", "nwb")
    return overlaps[0]["overlapping_type2_paths"]


def find_overlaps(df, type1, type2):
    # Prepare start/end columns
    df = df.copy()
    df["start"] = df["begin"]
    df["end"] = df["begin"] + df["duration"]

    df1 = df[df["type"] == type1]
    df2 = df[df["type"] == type2]

    results = []

    for _, row1 in df1.iterrows():
        overlaps = df2[(df2["start"] < row1["end"]) & (row1["start"] < df2["end"])]

        results.append(
            {
                "type1_path": row1["path"],
                "type1_start": row1["start"],
                "type1_end": row1["end"],
                "overlapping_type2_paths": overlaps["path"].tolist(),
                "overlapping_rows": overlaps,  # optional: keep full rows
            }
        )

    return results

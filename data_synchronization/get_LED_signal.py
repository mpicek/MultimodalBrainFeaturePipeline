import numpy as np
import os
import argparse
from scipy.ndimage import gaussian_filter
def smooth(y, box_pts):
    return gaussian_filter(y, box_pts)
from LED_utils import get_LED_signal_from_video

N_FRAMES_TO_COMPUTE_LED_STD_FROM = 9999999999999999999 # only 40 needed because we are only interested in the position of the LED, not the signal itself
DOWNSCALE_FACTOR = 2

def get_LED_signal(path_video, led_position, binary_mask):
    average_values = get_LED_signal_from_video(path_video, binary_mask, led_position, N_FRAMES_TO_COMPUTE_LED_STD_FROM, DOWNSCALE_FACTOR)
    return average_values


def video_folder_get_LED(video_folder, led_position_folder, output_folder, extension):

    extension = extension.lower()

    # create the output_folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed_files = os.listdir(output_folder)
    processed_video_files = [(os.path.basename(file)[:-len('_LED_signal.npy')] + extension).lower() for file in processed_files if file.endswith('_LED_signal.npy')]

    print("Looking for video files in folder:", video_folder)

    video_count = 0
    for root, _, files in os.walk(video_folder):
        for file in files:
            if not file.lower().endswith(extension):
                continue
            path_video = os.path.join(root, file)
            video_basename = os.path.splitext(os.path.basename(path_video))[0]

            if os.path.basename(path_video).lower() in processed_video_files:
                print(f"{path_video} already processed. Skipping")
                continue

            led_position_path = os.path.join(led_position_folder, video_basename + '_LED_position.npy')
            binary_mask_path = os.path.join(led_position_folder, video_basename + '_LED_binary_mask.npy')
            if not os.path.exists(led_position_path) or not os.path.exists(binary_mask_path):
                print(f"No LED position/mask found for {path_video}. Skipping.")
                continue

            video_count += 1
            print(f"Processing video {video_count}: {path_video}")
            try:
                led_position = np.load(led_position_path)
                binary_mask = np.load(binary_mask_path)
                led_signal = get_LED_signal(path_video, led_position, binary_mask)
                np.save(os.path.join(output_folder, video_basename + '_LED_signal.npy'), led_signal)
            except Exception:
                print(f"Error with file {path_video}")


def main(video_folder, led_position_folder, output_folder, extension):

    video_folder_get_LED(video_folder, led_position_folder, output_folder, extension)

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Extract the LED signals from video files.")
    parser.add_argument("video_folder", help="Path to the folder containing video files.")
    parser.add_argument("led_position_folder", help="Path to the folder containing led position .npy files.")
    parser.add_argument(
        "extension",
        type=lambda v: (v if v.startswith(".") else f".{v}").lower(),
        choices=[".mp4", ".mkv"],
        help="video extension (.mp4 or .mkv, case-insensitive)",
    )
    parser.add_argument("output_folder", help="Path to the output folder LED signals.")
    args = parser.parse_args()
    main(args.video_folder, args.led_position_folder, args.output_folder, args.extension)
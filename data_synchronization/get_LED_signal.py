import numpy as np
import os
import argparse
from scipy.ndimage import gaussian_filter
def smooth(y, box_pts):
    return gaussian_filter(y, box_pts)
from LED_utils import get_LED_signal_from_video
from LED_video_main import crop_subsampled_LED_red_channel_from_video_for_std

N_FRAMES_TO_COMPUTE_LED_STD_FROM = 9999999999999999999 # only 40 needed because we are only interested in the position of the LED, not the signal itself
DOWNSCALE_FACTOR = 2
DOWNSAMPLE_FRAMES_FACTOR = 15 # some blinks are very short, so we can't downsample too much

def get_LED_signal(path_video, led_position=None, binary_mask=None):


    # fallback in case we didn't have the led_position and binary_mask files (e.g. if we are processing a new video)
    # but it's recommended to do it separately to save time
    if led_position is None or binary_mask is None:
        subsampled_video_array, led_position, _ = crop_subsampled_LED_red_channel_from_video_for_std(
            path_video,
            100,
            DOWNSCALE_FACTOR,
            DOWNSAMPLE_FRAMES_FACTOR
        )

        binary_mask = np.full((subsampled_video_array.shape[1], subsampled_video_array.shape[2]), True)

    average_values = get_LED_signal_from_video(path_video, binary_mask, led_position, N_FRAMES_TO_COMPUTE_LED_STD_FROM, DOWNSCALE_FACTOR)

    return average_values


def video_folder_get_LED(video_folder, led_position_folder, output_folder, extension):

    extension = extension.lower()

    # create the output_folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed_files = os.listdir(output_folder)
    processed_video_files = [(os.path.basename(file)[:-len('_LED_signal.npy')] + extension).lower() for file in processed_files if file.endswith('_LED_signal.npy')]

    for root, _, files in os.walk(video_folder):
        for file in files:
            if file.lower().endswith(extension):
                path_video = os.path.join(root, file)
                video_basename = os.path.splitext(os.path.basename(path_video))[0]

                if os.path.basename(path_video).lower() in processed_video_files:
                    print(f"File {path_video} already processed. Skipping")
                    continue

                print("Processing file:", path_video)
                try: # this fails if there is no led_position and binary_mask files => we have to select it manually now
                    led_position = np.load(os.path.join(led_position_folder, video_basename + '_LED_position.npy'))
                    binary_mask = np.load(os.path.join(led_position_folder, video_basename + '_LED_binary_mask.npy'))
                    led_signal = get_LED_signal(path_video, led_position, binary_mask)
                except Exception as e:
                    led_signal = get_LED_signal(path_video)

                np.save(os.path.join(output_folder, video_basename + '_LED_signal.npy'), led_signal)


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
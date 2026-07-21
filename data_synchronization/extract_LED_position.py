import numpy as np
import os
import argparse
from scipy.ndimage import gaussian_filter
def smooth(y, box_pts):
    return gaussian_filter(y, box_pts)
from LED_video_main import crop_subsampled_LED_red_channel_from_video_for_std

N_FRAMES_TO_COMPUTE_LED_STD_FROM = 9000
DOWNSCALE_FACTOR = 2
DOWNSAMPLE_FRAMES_FACTOR = 15 # some blinks are very short, so we can't downsample too much

def extract_led_position_folder(video_folder, output_folder, n_frames, downscale_factor, downsample_frames_factor, extension):

    processed_files = os.listdir(output_folder)
    processed_video_files = [os.path.basename(file)[:-len('_LED_position.npy')] + extension for file in processed_files if file.endswith('_LED_position.npy')]

    print("Looking for video files in folder:", video_folder)
    video_paths = []
    for root, _, files in os.walk(video_folder):
        for file in files:
            if file.endswith(extension):
                video_paths.append(os.path.join(root, file))

    total_videos = len(video_paths)

    for i, path_video in enumerate(video_paths, start=1):
        if os.path.basename(path_video) in processed_video_files:
            print(f"Processing: {i}/{total_videos} - {path_video} already processed. Skipping")
            continue
        print(f"Processing: {i}/{total_videos} - {path_video}")
        try:
            subsampled_video_array, ref_point, _ = crop_subsampled_LED_red_channel_from_video_for_std(
                    path_video,
                    n_frames,
                    downscale_factor,
                    downsample_frames_factor
                )

            binary_mask = np.full((subsampled_video_array.shape[1], subsampled_video_array.shape[2]),True,)

            video_basename = os.path.splitext(os.path.basename(path_video))[0]
            np.save(os.path.join(output_folder, video_basename + "_LED_position.npy"),ref_point,)
            np.save(os.path.join(output_folder, video_basename + '_LED_binary_mask.npy'),binary_mask,)
        except:
            print(f"Error with file {path_video}")


def main(video_folder, output_folder, extension):



    extract_led_position_folder(
        video_folder, output_folder, N_FRAMES_TO_COMPUTE_LED_STD_FROM, DOWNSCALE_FACTOR, N_FRAMES_TO_COMPUTE_LED_STD_FROM, extension
    )


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Extract the LED signals from video files.")
    parser.add_argument("video_folder", help="Path to the folder containing video files.")
    parser.add_argument("output_folder", help="Where to put output.")
    parser.add_argument(
        "extension",
        type=lambda v: v if v.startswith(".") else f".{v}",
        choices=[".MP4", ".mkv"],
        help="video extension (.MP4 or .mkv)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    main(args.video_folder, args.output_folder, args.extension)

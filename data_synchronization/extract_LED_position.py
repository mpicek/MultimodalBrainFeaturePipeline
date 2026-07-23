import numpy as np
import os
import argparse
from scipy.ndimage import gaussian_filter
def smooth(y, box_pts):
    return gaussian_filter(y, box_pts)
from LED_video_main import get_single_frame_for_led_cropping

DOWNSCALE_FACTOR = 2
FRAMES_TO_SKIP_AT_START = 30  # skip the first few frames, which are often under-exposed/unstable

def extract_led_position_folder(video_folder, output_folder, downscale_factor, extension):

    extension = extension.lower()

    processed_files = os.listdir(output_folder)
    processed_video_files = [(os.path.basename(file)[:-len('_LED_position.npy')] + extension).lower() for file in processed_files if file.endswith('_LED_position.npy')]

    print("Looking for video files in folder:", video_folder)
    video_paths = []
    for root, _, files in os.walk(video_folder):
        for file in files:
            if file.lower().endswith(extension):
                video_paths.append(os.path.join(root, file))

    total_videos = len(video_paths)

    for i, path_video in enumerate(video_paths, start=1):
        if os.path.basename(path_video).lower() in processed_video_files:
            print(f"Processing: {i}/{total_videos} - {path_video} already processed. Skipping")
            continue
        print(f"Processing: {i}/{total_videos} - {path_video}")
        try:
            ref_point, cropped_LED_image_colorful = get_single_frame_for_led_cropping(
                    path_video,
                    skip_frames=FRAMES_TO_SKIP_AT_START,
                )

            mask_shape = cropped_LED_image_colorful[..., 2][::downscale_factor, ::downscale_factor].shape
            binary_mask = np.full(mask_shape, True)

            video_basename = os.path.splitext(os.path.basename(path_video))[0]
            np.save(os.path.join(output_folder, video_basename + "_LED_position.npy"),ref_point,)
            np.save(os.path.join(output_folder, video_basename + '_LED_binary_mask.npy'),binary_mask,)
        except Exception:
            print(f"Error with file {path_video}")


def main(video_folder, output_folder, extension):

    extract_led_position_folder(
        video_folder, output_folder, DOWNSCALE_FACTOR, extension
    )


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Extract the LED signals from video files.")
    parser.add_argument("video_folder", help="Path to the folder containing video files.")
    parser.add_argument("output_folder", help="Where to put output.")
    parser.add_argument(
        "extension",
        type=lambda v: (v if v.startswith(".") else f".{v}").lower(),
        choices=[".mp4", ".mkv"],
        help="video extension (.mp4 or .mkv, case-insensitive)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    main(args.video_folder, args.output_folder, args.extension)

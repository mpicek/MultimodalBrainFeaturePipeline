import numpy as np
import cv2
import os
import argparse
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
def smooth(y, box_pts):
    return gaussian_filter(y, box_pts)
from wisci_utils import get_LED_data
from LED_utils import get_LED_signal_from_video, get_LED_mask, get_LED_signal_from_video
from LED_video_main import crop_subsampled_LED_red_channel_from_video_for_std
from LED_GUI_cropper import ImageCropper
from LED_video_main import crop_subsampled_LED_red_channel_from_video_for_std


def extract_led_position_folder(mp4_folder, downscale_factor, downsample_frames_factor):

    processed_files = os.listdir(mp4_folder)
    processed_mp4_files = [os.path.basename(file)[:-len('_LED_position.npy')] + '.mp4' for file in processed_files if file.endswith('_LED_position.npy')]

    for root, _, files in os.walk(mp4_folder):
        for file in files:
            if file.endswith('.mp4'):
                path_mp4 = os.path.join(root, file)

                if os.path.basename(path_mp4) in processed_mp4_files:
                    print(f"File {path_mp4} already processed. Skipping")
                    continue

                subsampled_video_array, ref_point, _ = crop_subsampled_LED_red_channel_from_video_for_std(
                    path_mp4,
                    10,
                    downscale_factor,
                    downsample_frames_factor
                )

                binary_mask = np.full((subsampled_video_array.shape[1], subsampled_video_array.shape[2]), True)

                mp4_basename = os.path.basename(path_mp4)[:-4]
                np.save(os.path.join(mp4_folder, mp4_basename + '_LED_position.npy'), ref_point)
                np.save(os.path.join(mp4_folder, mp4_basename + '_LED_binary_mask.npy'), binary_mask)


def main(mp4_folder):

    # if we want to process just one file (in that case args.mp4_folder has to be a path to the file)
    # average_values = mp4_get_LED_signal(args.mp4_folder)
    # import matplotlib.pyplot as plt
    # plt.plot(np.arange(len(average_values)) / 25, average_values)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Average pixel value')
    # plt.title('Average pixel value over time')
    # plt.show()
    downscale_factor = 2
    downsample_frames_factor = 1

    extract_led_position_folder(mp4_folder, downscale_factor, downsample_frames_factor)

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Extract the LED signals from .mp4 files.")
    parser.add_argument("mp4_folder", help="Path to the folder containing mp4 files.")
    args = parser.parse_args()
    main(args.mp4_folder)
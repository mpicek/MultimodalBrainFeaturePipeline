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
from accelerometer import get_LED_data
from LED_utils import get_LED_signal_from_video
from LED_video_main import crop_subsampled_LED_red_channel_from_video_for_std


def mp4_get_LED_signal(path_mp4):

    n_frames = 9999999999999999999
    downscale_factor = 2
    downsample_frames_factor = 1

    subsampled_video_array, ref_point, _ = crop_subsampled_LED_red_channel_from_video_for_std(
        path_mp4,
        100,
        downscale_factor,
        downsample_frames_factor
    )

    binary_mask = np.full((subsampled_video_array.shape[1], subsampled_video_array.shape[2]), True)

    average_values = get_LED_signal_from_video(path_mp4, binary_mask, ref_point, n_frames, downscale_factor)

    return average_values


def mp4_folder_get_LED(mp4_folder, output_folder):

    # create the output_folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(mp4_folder):
        for file in files:
            if file.endswith('.mp4'):
                path_mp4 = os.path.join(root, file)

                led_signal = mp4_get_LED_signal(path_mp4)
                mp4_basename = os.path.basename(path_mp4)[:-4]
                np.save(os.path.join(output_folder, mp4_basename + '_LED_signal.npy'), led_signal)


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Extract the LED signals from .mp4 files.")
    parser.add_argument("mp4_folder", help="Path to the folder containing mp4 files.")
    parser.add_argument("output_folder", help="Path to the output folder LED signals.")
    args = parser.parse_args()

    # if we want to process just one file (in that case args.mp4_folder has to be a path to the file)
    # average_values = mp4_get_LED_signal(args.mp4_folder)
    # import matplotlib.pyplot as plt
    # plt.plot(np.arange(len(average_values)) / 25, average_values)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Average pixel value')
    # plt.title('Average pixel value over time')
    # plt.show()

    mp4_folder_get_LED(args.mp4_folder, args.output_folder)

if __name__ == "__main__":
    main()
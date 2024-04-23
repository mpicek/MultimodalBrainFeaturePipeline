##################################################################################
# Does not log the cropping process
# Crops only videos that have corresponding bounding box data
# Skips already cropped videos
##################################################################################

import os
import subprocess
import numpy as np
import argparse

def crop_videos(mp4_folder, extraction_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over videos in mp4_folder
    for video_file in os.listdir(mp4_folder):
        if video_file.endswith(".mp4"):
            mp4_filename = video_file[:-4]
            npy_file = os.path.join(extraction_folder, f"{mp4_filename}_bounding_box.npy")

            # Check if corresponding numpy file exists
            if os.path.exists(npy_file):
                # Read bounding box coordinates from numpy file
                bounding_box = np.load(npy_file)

                width = bounding_box[2] - bounding_box[0]
                height = bounding_box[3] - bounding_box[1]

                # Crop video using ffmpeg
                input_path = os.path.join(mp4_folder, video_file)
                output_path = os.path.join(output_folder, f"{mp4_filename}_cropped.mp4")
                
                # proceed if the file does not exist
                if os.path.exists(output_path):
                    print(f"Video {video_file} already cropped. Skipping.")
                    continue
                command = (f"ffmpeg -i {input_path} -loglevel warning -vf "
                           f"crop={width}:{height}:{bounding_box[0]}:{bounding_box[1]} "
                           f"{output_path}")
                subprocess.run(command, shell=True)
                print(f"Video {video_file} cropped and saved as {output_path}")
            else:
                print(f"No bounding box data found for {video_file}")

# Example usage
mp4_folder = "/home/vita-w11/mpicek/processed/data/mp4_0"
extraction_folder = "/home/vita-w11/mpicek/processed/data/extraction_0"
output_folder = "/home/vita-w11/mpicek/processed/data/cropped"

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Extract the LED signals from .mp4 files.")
    parser.add_argument("mp4_folder", help="Path to the folder containing mp4 files.")
    parser.add_argument("extraction_folder", help="Path to the folder containing bounding boxes (.npy files).")
    parser.add_argument("output_folder", help="Path to the output folder LED signals.")
    args = parser.parse_args()

    crop_videos(args.mp4_folder, args.extraction_folder, args.output_folder)
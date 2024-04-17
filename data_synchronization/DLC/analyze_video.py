import argparse
import deeplabcut
import glob
import os

def analyze_videos(mp4_folder, output_folder, config_path):
    video_files = glob.glob(os.path.join(mp4_folder, '*.mp4'))
    print(video_files)
    deeplabcut.analyze_videos(config_path, video_files, destfolder=output_folder, shuffle=1, save_as_csv=True, videotype='mp4', gputouse=1)

def main():
    parser = argparse.ArgumentParser(description="Analyze videos with DeepLabCut.")
    parser.add_argument("mp4_folder", help="Path to the folder containing .mp4 files.")
    parser.add_argument("output_folder", help="Where to put output.")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    config_path = '/mpicek/dlc_nose_detector/UP2_movement_sync-mpicek-2024-04-15/config.yaml'
    print(f"THE FOLLOWING CONFIG WILL BE USED:")
    print(config_path)

    analyze_videos(args.mp4_folder, args.output_folder, config_path)

if __name__ == "__main__":
    main()
import argparse
import deeplabcut
import glob
import os

def analyze_videos(mp4_folder, config_path):
    video_files = glob.glob(os.path.join(mp4_folder, '*.mp4'))
    print(video_files)
    deeplabcut.analyze_videos(config_path, video_files, shuffle=1, save_as_csv=True, videotype='mp4', gputouse=1)

def main():
    parser = argparse.ArgumentParser(description="Analyze videos with DeepLabCut.")
    parser.add_argument("mp4_folder", help="Path to the folder containing .mp4 files.")
    args = parser.parse_args()

    config_path = '/data/UP2_synchronization_videos_concatenated-mpicek-2024-03-20_2/config.yaml'
    print(f"THE FOLLOWING CONFIG WILL BE USED:")
    print(config_path)

    analyze_videos(args.mp4_folder, config_path)

if __name__ == "__main__":
    main()
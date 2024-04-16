import sys
sys.path.append('/repo/data_synchronization')
import ffmpegio
import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import subprocess
import shutil
from mediapipe_utils import *
from synchronization_utils import *
import argparse
import pandas as pd

def create_videos_from_bag(path_bag, output_folder):
    """
    Extracts a vector representing face movement over time from a realsense .bag file.

    This function processes a the .bag file specified by `path_bag` to detect and track
    a bellow nose to forehead vector using a Mediapipe face model.

    Parameters:
    - path_bag (str): Path to the .bag file containing the realsense data.
    - mediapipe_face_model_file (str): Path to the Mediapipe face model file for face detection.

    Returns:
    - numpy.ndarray: An array of shape (frames, 3)
      Each row contains the x, y, z vector direction relative to another facial point.

    Notes:
    - The function assumes a video resolution of 640x480.
    - RealSense SDK is used for video processing, and playback is adjusted to process frames
      as needed, avoiding real-time constraints.
    - Frames where the face is not detected are handled by repeating the last valid detection.
    - The function visualizes face landmarks and their movements in real-time during processing.
    - Press 'q' to quit the visualization and processing early.
    - The function smooths the movement data for visualization purposes.

    Raises:
    - Exception: If there are issues initializing the video pipeline or processing frames.
    """

    img_height = 480
    img_width = 640

    try:

        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, path_bag)

        config.enable_stream(rs.stream.depth, img_width, img_height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, img_width, img_height, rs.format.bgr8, 30)

        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(True)
        duration = playback.get_duration().total_seconds() * 1000
        print(f"Overall video duration: {playback.get_duration()}")
    except Exception as e:
        return 0, str(e), 0, 0
    
    # we want to set_real_time to False, so that we can process the frames as slowly as necessary.
    # However, there are errors reading it at the beginning and at the end (hopefully not in the middle)
    # when we set it to False right at the beginning. Therefore, we set it to True at the beginning 
    # and then to False after the first frame is read. The end is handled differently - with reading the frames
    # up to the duration of the video - 1s (to be sure that we don't miss the last frame)

    first_frame = True
    prev_ts = -1
    max_frame_nb = 0
    frame_nb = -1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) 
    ts = 0
    time_series = []
    frames_list = []
    video_counter = 0
    failed = False
    problematic_frames = 0
    # playback.seek(datetime.timedelta(seconds=73*4))
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except Exception as e:
                print("READING ERROR")
                duration = prev_ts # use previous timestamp as the current one can be corrupt
                print(f"Duration after reading error: {duration // (1000*60)}:{(duration // 1000) % 60}:{(duration % 1000) // 10}")
                failed = str(e)
                break


            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            playback.set_real_time(False)

            # skipped frame by RealSense
            if not depth_frame or not color_frame:
                problematic_frames += 1
                if len(frames_list) > 0:
                    frames_list.append(frames_list[-1])
                else:
                    # the detection will not work, which is ok, we filter it with quality
                    frames_list.append(np.zeros((480, 640, 3), dtype=np.uint8))
                continue

            frame_nb = color_frame.get_frame_number()

            # finish of the file (no more new frames)
            if frame_nb < max_frame_nb:
                break

            max_frame_nb = frame_nb

            ts = frames.get_timestamp()


            depth_image_rs = np.asanyarray(depth_frame.get_data())
            color_image_rs = np.asanyarray(color_frame.get_data()) # returns RGB!


            if first_frame: 
                t0 = ts
                first_frame = False
            
            # the video is at the end (without the last second) so we kill the reading
            # (there was an error with the last frame, this handles it)
            if ts - t0 + 1000 > duration:
                break

            if prev_ts > int(ts-t0):
                problematic_frames += 1
                continue

            if prev_ts == int(ts-t0):
                problematic_frames += 1
                continue


            frames_list.append(cv2.cvtColor(color_image_rs, cv2.COLOR_BGR2RGB))
            time_series.append(ts)
            
            prev_prev_ts = prev_ts
            prev_ts = int(ts-t0)

            if prev_ts - prev_prev_ts > 2*int(1000/30) - 10:
                num_of_skipped_frames = (prev_ts - prev_prev_ts) // int(1000/30)
                problematic_frames += num_of_skipped_frames

                if len(frames_list) > 0:
                    for i in range(num_of_skipped_frames):
                        frames_list.append(frames_list[-1])
                else:
                    for i in range(num_of_skipped_frames):
                        frames_list.append(np.zeros((480, 640, 3), dtype=np.uint8))

            ch = cv2.waitKey(1)
            if ch==113: #q pressed
                failed = "Keyboard Interrupt"
                break


            if len(time_series) % (30*30) == 0:
                formatted_number = '{:04d}'.format(video_counter)
                frames_list = np.stack(frames_list)
                non_zero_index = np.where((frames_list != np.zeros((480, 640, 3))).any(axis=1))[0][0]
                frames_list[:non_zero_index] = frames_list[non_zero_index]
                ffmpegio.video.write(os.path.join(output_folder, formatted_number + '.mp4'), 30, frames_list, show_log=True, loglevel="warning")
                print(f"{len(time_series) / (30*30) / 2 + video_counter/2} minutes processed")
                np.save(os.path.join(output_folder, formatted_number + '.npy'), np.array(time_series))
                time_series = []
                frames_list = []
                video_counter += 1

            t = ts - t0
    except Exception as e:
        failed = str(e)
        duration = prev_ts

    finally:
        if len(frames_list) > 0:
            frames_list = np.stack(frames_list)
            non_zero_index = np.where((frames_list != np.zeros((480, 640, 3))).any(axis=1))[0][0]
            frames_list[:non_zero_index] = frames_list[non_zero_index]
            formatted_number = '{:04d}'.format(video_counter)
            ffmpegio.video.write(os.path.join(output_folder, formatted_number + '.mp4'), 30, frames_list, show_log=True, loglevel="warning")
            print(f"{len(time_series) / (30*30) / 2 + video_counter/2} minutes processed")
            np.save(os.path.join(output_folder, formatted_number + '.npy'), np.array(time_series))
            time_series = []

        pipeline.stop()
        cv2.destroyAllWindows()
    
    return prev_ts / 1000, failed, problematic_frames


def bag2mp4(path_bag, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tmp_folder = os.path.join(output_folder, 'tmp')
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)

    duration, failed, problematic_frames = create_videos_from_bag(path_bag, tmp_folder)
    np.save(os.path.join(output_folder, os.path.basename(path_bag)[:-4] + '_duration.npy'), duration)

    # List all mp4 files in the current directory alphabetically
    mp4_files = sorted([os.path.join(tmp_folder, file) for file in os.listdir(tmp_folder) if file.endswith('.mp4')])

    timestamp_files = []

    if len(mp4_files) > 0:
        # Create a temporary file listing all mp4 files
        with open(os.path.join(tmp_folder, 'file_list.txt'), 'w') as file_list:
            for mp4_file in mp4_files:
                file_list.write(f"file '{mp4_file}'\n")
                timestamp_files.append(mp4_file[:-4] + '.npy')
        
        concatenated_timestamps = np.concatenate([np.load(file) for file in timestamp_files])
        np.save(os.path.join(output_folder, os.path.basename(path_bag)[:-4] + '_timestamps.npy'), concatenated_timestamps)


        # Concatenate mp4 files using ffmpeg
        output_file_name = os.path.join(output_folder, os.path.basename(path_bag)[:-4] + '.mp4')
        print(f"The video is being saved into: {output_file_name}")
        # -y flag is used to overwrite the output file if it already exists
        # -f concat flag is used to concatenate the files
        # -c copy flag is used to copy the streams without re-encoding
        # -loglevel error flag is used to suppress the ffmpeg output (otherwise it's super annoying and extremely verbose)
        try:
            subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', os.path.join(tmp_folder, 'file_list.txt'), '-c', 'copy', output_file_name, '-loglevel', 'error'])
        except:
            if failed == False:
                failed = "Subprocess ffmpeg failed."
            else:
                failed += "... AND ALSO SUBPROCESS FFMPEG FAILED!."

    # Delete the original mp4 files in the tmp folder
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    
    return duration, failed, problematic_frames

def bag_folder2mp4(bag_folder, output_folder, log_table_path):
    """
    Convert all .bag files in a given folder to .mp4 files and update a log table (append rows if the table exists).

    Args:
        bag_folder (str): The path to the folder containing the .bag files.
        output_folder (str): The path to the folder where the .mp4 files will be saved.
        log_table_path (str): The path to the log table CSV file.
    """
    
    if os.path.exists(log_table_path):
        log_df = pd.read_csv(log_table_path)
    else:
        log_df = pd.DataFrame(columns=['bag_path', 'failed', 'duration', 'problematic_frames', 'mp4_name'])

    num_rows = len(log_df)

    file_counter = 0 + num_rows
    for root, _, files in os.walk(bag_folder):
        for file in files:
            if file.endswith('.bag'):
                bag_path = os.path.join(root, file)

                # Skip the file if it has already been processed (based on only basename, not the full path)
                # If so, skip the file
                all_processed_files = log_df['bag_path'].values
                all_processed_files = [os.path.basename(file) for file in all_processed_files]
                if os.path.basename(bag_path) in all_processed_files:
                    print(f"Skipping {bag_path} as it has already been processed.")
                    continue

                duration, failed, problematic_frames = bag2mp4(bag_path, output_folder)
                log_df.loc[file_counter] = [
                    bag_path,
                    failed,
                    duration,
                    problematic_frames,
                    os.path.basename(bag_path)[:-4] + '.mp4'
                ]
                file_counter += 1
                log_df.to_csv(log_table_path, index=False)


    log_df.to_csv(log_table_path, index=False)
    print(f"Log table saved to {log_table_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert bag files to mp4 and log the status.")
    parser.add_argument("bag_folder", type=str, help="Path to the folder containing bag files.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder for mp4 files.")
    parser.add_argument("log_table_path", type=str, help="Path to save the log table as a CSV file.")
    args = parser.parse_args()

    ##########################
    # IF WE WANT TO PROCESS JUST ONE FILE (IN THAT CASE args.bag_folder HAS TO BE A PATH TO THE FILE)
    # failed = bag2mp4(args.bag_folder, args.output_folder)
    # print(failed)
    ##########################

    bag_folder2mp4(args.bag_folder, args.output_folder, args.log_table_path)
    # python bag2mp4.py --bag_folder /home/vita-w11/Downloads/bags/ --output_folder /home/vita-w11/Downloads/bags/output_test --log_table_path /home/vita-w11/Downloads/bags/output_test/log.csv

if __name__ == "__main__":
    main()
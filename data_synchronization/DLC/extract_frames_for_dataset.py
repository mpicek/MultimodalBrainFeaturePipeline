import os
import numpy as np
import os
import pyrealsense2 as rs
import cv2
import ffmpegio
import argparse


class Exception5000(Exception):
    pass


def extract_frames(path_bag, output_path, how_often=1000, end_frame=4000, visualize=False):
    """
    Extracts a vector representing face movement over time from a realsense .bag file.

    This function processes a the .bag file specified by `path_bag` to detect and track
    a bellow nose to forehead vector using a Mediapipe face model.

    Parameters:
    - path_bag (str): Path to the .bag file containing the realsense data.

    Returns:
    tuple: forehead_points, quality, face2cam, cam2face, face_coordinates_origin, duration
        - forehead_points (numpy.ndarray): An array of shape (frames, 3)
            Each row contains the x, y, z coordinates (in camera space) of a point on the forehead.
            The magnitude should be normalized so that all three coordinates have the same scale (like in 3d).
            (because Mediapipe returns the coordinates in the range [0, 1] relative to the
            height and width of the image, so we multiplied it by the width and height of the image)
        - quality (numpy.ndarray): An array of shape (frames,) that is 1 if mediapipe detected
            the face and 0 otherwise. This is used to indicate the quality of the forehead_points.
        - face2cam (numpy.ndarray): A 3x3 matrix representing the transformation from face to camera coordinates.
        - cam2face (numpy.ndarray): A 3x3 matrix representing the transformation from camera to face coordinates.
        - face_coordinates_origin (numpy.ndarray): A 3x1 vector representing the origin of the face coordinate system.
            Expressed in the camera coordinate system.
        - duration (float): The duration of the video in seconds.

    Beginning of the processing:
        1) Detect the patient's face (with face_recognition library) - this region will be used in every frame
        (we assume the patient doesn't move that much)
        2) Construct the face coordinate system (with mediapipe library)

    Notes:
    - The function assumes a video resolution of 640x480.
    - Frames where the face is not detected are handled by repeating the last valid detection.
    - Frames not detected/or some error with face detection/mediapipe AT THE BEGINNING are handled by
    - copying the first valid value to the beginning (so that there is no big jump in the positions)
        - it's first set to [0, 0, 0], at the end we rewrite it to the first valid value
    - The function visualizes face landmarks and their movements in real-time during processing.
    - Press 'q' to quit the visualization and processing early.

    Possible errors:
    - missed frame (by Realsense - detectable by timestamp) - HANDLED
    - repeated frame (by Realsense - detectable by timestamp) - HANDLED
    - face detection doesn't detect anything AT THE BEGINNING FOR FACE COORDINATE SYSTEM - HANDLED
    - mediapipe doesn't detect anything AT THE BEGINNING FOR FACE COORDINATE SYSTEM - HANDLED
    - mediapipe doesn't detect anything (just regular extraction of forehead points) - HANDLED
    - frames don't arrive within 5000ms - NOT-HANDLED (TODO)
        - this is handled only at the beginning of the video (that's why we set real time to True and then to False in the first iteration)
    - weird error at the end (thus I finish 1s before the end) - HANDLED

    Raises:
    - Exception5000: If frames don't arrive within 5000ms.
    """

    img_height, img_width = 480, 640

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, path_bag)
    config.enable_stream(rs.stream.depth, img_width, img_height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, img_width, img_height, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    
    # we want to set_real_time to False, so that we can process the frames as slowly as necessary.
    # However, there are errors reading it at the beginning and at the end (hopefully not in the middle)
    # when we set it to False right at the beginning. Therefore, we set it to True at the beginning 
    # and then to False after the first frame is read. The end is handled differently - with reading the frames
    # up to the duration of the video - 1s (to be sure that we don't miss the last frame)
    playback.set_real_time(True)

    first_frame = True
    ts = 0
    prev_ts = -1
    max_frame_nb = 0
    frame_nb = -1

    duration = playback.get_duration().total_seconds() * 1000
    print(f"Overall video duration: {playback.get_duration()}")

    frame_list = []

    # playback.seek(datetime.timedelta(seconds=73*4))
    i = 0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except:
                print("READING ERROR")
                raise Exception5000()


            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            playback.set_real_time(False)

            # skipped frame by RealSense
            if not depth_frame or not color_frame:
                continue

            frame_nb = color_frame.get_frame_number()

            # end of the file (no more new frames)
            if frame_nb < max_frame_nb:
                break

            max_frame_nb = frame_nb

            ts = frames.get_timestamp()

            color_image_rs = np.asanyarray(color_frame.get_data()) # returns RGB!

            i += 1

            if i%how_often == 0:
                frame_list.append(cv2.cvtColor(color_image_rs, cv2.COLOR_BGR2RGB))

            if i > end_frame:
                break

            if first_frame: 
                t0 = ts
                first_frame = False
            
            # the video is at the end (without the last second) so we kill the reading
            # (there was an error with the last frame, this handles it)
            if ts - t0 + 1000 > duration:
                duration -= 1000 # for further processing we need to remember that we finished 1s earlier
                break

            if prev_ts >= int(ts-t0):
                # doubled frame or some other error in ordering (we don't include the frame
                # as we don't want it multiple times)
                continue

            prev_prev_ts = prev_ts
            prev_ts = int(ts-t0)

            ch = cv2.waitKey(1)
            if ch==113: #q pressed
                break

            t = ts - t0
    finally:
        frame_list = np.stack(frame_list)
        bag_filename = os.path.basename(path_bag)
        output_file = os.path.join(output_path, f"{bag_filename}_frame_{frame_nb}.mp4")
        ffmpegio.video.write(output_file, 3, frame_list, show_log=True)
        pipeline.stop()


# if name == main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from .bag files.')
    parser.add_argument('--folder_path', type=str, help='Path to the folder containing .bag files')
    parser.add_argument('--output_path', type=str, help='Path to the output folder')
    args = parser.parse_args()

    folder_path = args.folder_path
    output_path = args.output_path

    # Make output_path if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Get all .bag files in the folder
    bag_files = [file for file in os.listdir(folder_path) if file.endswith(".bag")]

    for bag_file in bag_files:
        bag_file_path = os.path.join(folder_path, bag_file)

        # extract 1 frame every 2s (and extract the first 4 minutes)
        extract_frames(bag_file_path, output_path, how_often=120, end_frame=2*60*30, visualize=True)
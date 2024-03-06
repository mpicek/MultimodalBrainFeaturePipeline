import pandas as pd
import scipy.io
import pyrealsense2 as rs
import numpy as np
import cv2
import face_recognition
import matplotlib.pyplot as plt
import mediapipe as mp
from IPython.display import clear_output

from mediapipe_utils import *
from signal_utils import *

def extract_face_vector(path_bag, mediapipe_face_model_file):
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

    detector = setup_mediapipe_face(mediapipe_face_model_file)
    img_height = 480
    img_width = 640

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
    prev_ts = -1
    max_frame_nb = 0
    frame_nb = -1
    around_face_factor = 1
    face_location = None

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) 
    ts = 0

    time_series = []
    nose_stacked = []
    duration = playback.get_duration().total_seconds() * 1000
    print(f"Overall video duration: {playback.get_duration()}")
    # playback.seek(datetime.timedelta(seconds=73*4))
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except:
                print("READING ERROR")
                playback.set_real_time(True)


            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            playback.set_real_time(False)


            # skipped frame by RealSense
            if not depth_frame or not color_frame:
                print("SKIPPED FRAME! HANDLE IT!")
                continue

            last_frame_nb = frame_nb
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

            if prev_ts >= int(ts-t0):
                # doubled frame or some other error in ordering (we don't include the frame
                # as we don't want it multiple times)
                continue

            if face_location is None:
                # convert it to BGR as face_recognition uses BGR
                face_locations = face_recognition.face_locations(cv2.cvtColor(color_image_rs, cv2.COLOR_RGB2BGR))
                for face_location in face_locations:

                    top, right, bottom, left = face_location
                    face_width = right - left
                    face_height = bottom - top

            # Crop the face
            # we cut the face with some margin around it (defined by around_face_factor * face_width or face_height)
            # but we have to be careful not to go out of the image (that's why all the max and min functions)                    
            face_image = color_image_rs[
                max(0, top - int(face_height * around_face_factor)):min(img_height, int(bottom + face_height * around_face_factor)),
                max(0, left - int(face_width * around_face_factor)):min(img_width, int(right + face_width * around_face_factor))
            ]

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            detection_result = detector.detect(image)
            try:
                nose_stacked.append(np.array([
                    detection_result.face_landmarks[0][8].x - detection_result.face_landmarks[0][2].x,
                    detection_result.face_landmarks[0][8].y - detection_result.face_landmarks[0][2].y,
                    detection_result.face_landmarks[0][8].z - detection_result.face_landmarks[0][2].z,
                ]))
            except:
                print("Mediapipe didn't detect a thing! Using the last valid values.")
                if len(nose_stacked) > 0:
                    nose_stacked.append(nose_stacked[-1])

            # STEP 5: Process the detection result. In this case, visualize it.
            annotated_image = draw_face_landmarks_on_image(image.numpy_view(), detection_result)
            cv2.imshow("Face features", cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

            prev_prev_ts = prev_ts
            prev_ts = int(ts-t0)

            if prev_ts - prev_prev_ts > 2*int(1000/30) - 10:
                print(f"Skipped frame. Previous ts: {prev_ts}, difference {prev_ts - prev_prev_ts}")

            time_series.append([prev_ts])

            ch = cv2.waitKey(1)
            if ch==113: #q pressed
                break

            if len(time_series) % (60*30) == 0:
                print(f"{len(time_series) / (60*30)} minutes processed")

            t = ts - t0

            if frame_nb > 100:
                ax.cla()
                indices = np.arange(len(nose_stacked))
                x = smooth(np.gradient(np.array(nose_stacked)[:, 0]), 20)
                y = smooth(np.gradient(np.array(nose_stacked)[:, 1]), 20)
                z = smooth(np.gradient(np.array(nose_stacked)[:, 2]), 20)
                ax.plot(indices, x, label='x')
                ax.plot(indices, y, label='y')
                ax.plot(indices, z, label='z')
                clear_output(wait = True)
                plt.pause(0.00000001)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        nose_stacked = np.stack(nose_stacked)
    
    return nose_stacked

if __name__ == '__main__':
    path_bag = "/home/mpicek/repos/master_project/test_data/corresponding/cam0_911222060374_record_13_11_2023_1330_19.bag"
    # path_bag = "/home/mpicek/repos/master_project/test_data/corresponding/cam0_911222060374_record_13_11_2023_1337_20.bag"
    nose_vector_data = extract_face_vector(path_bag, mediapipe_face_model_file = '/home/mpicek/Downloads/face_landmarker.task')
    output_filename = 'nose_vector.npy'
    # print(f"Saving data of shape {nose_vector_data.shape} to {output_filename}")
    # np.save(output_filename, nose_vector_data)
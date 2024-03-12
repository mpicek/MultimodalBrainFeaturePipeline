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
    forehead_points, quality = [], []
    duration = playback.get_duration().total_seconds() * 1000
    print(f"Overall video duration: {playback.get_duration()}")

    # playback.seek(datetime.timedelta(seconds=73*4))

    face_coordinate_system = None

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
            print("SKIPPED FRAME BY THE REALSENSE!")
            if len(forehead_points) > 0:
                forehead_points.append(forehead_points[-1])
                quality.append(0)
            else:
                raise Exception("Skipped frame by RealSense at the beginning of the .bag file. We need to account for that in the length of the video (variable duration) (for FPS compuatation later).")
            continue

        frame_nb = color_frame.get_frame_number()

        # finish of the file (no more new frames)
        if frame_nb < max_frame_nb:
            break

        max_frame_nb = frame_nb

        ts = frames.get_timestamp()

        # TODO; process also the depth image
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
            if len(face_locations) == 0:
                raise Exception("No face detected in the first frame.")
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
            # Detect the point on forehead (and z is relative, but we don't care about it because a person
            # moves the head in an "angular" way and not back and forth. And the point is on the forehead,
            # so the angle of the head is going to be very similar to the z coordinate. It's an approximation
            # that should be good enough for our purposes)
            # We need to normalize the coordinates because mediapipe returns just the relative coordinates to the
            # size of the image. So we multiply it by the width and height of the image (cropped area only in our case
            # as mediapipe operates only on the cropped image!!). And, according to mediapipe docs, the z coordinate
            # is approximately of the same magnitude as x
            forehead_points.append(np.array([
                detection_result.face_landmarks[0][8].x * color_image_rs.shape[1],
                detection_result.face_landmarks[0][8].y * color_image_rs.shape[0],
                detection_result.face_landmarks[0][8].z * color_image_rs.shape[1],
            ]))

            quality.append(1)

            if face_coordinate_system is None:
                # We find the face coordinate system in the first detected frame (this will be the reference)
                # It's gonna be in matrix [x, y, z] where x, y, z are the bases of the coordinate system
                # they are normalized. X and Y don't have to be necessarily orthogonal, but we assume they are
                # z is orthogonal to x and y

                # it's a nose point :)
                face_coordinate_system = True

                # mediapipe returns y going down, we want it to go up (thus the minus sign in y coordinate)
                face_coordinates_origin = np.array([
                    detection_result.face_landmarks[0][1].x * color_image_rs.shape[1],
                    -detection_result.face_landmarks[0][1].y * color_image_rs.shape[0],
                    detection_result.face_landmarks[0][1].z * color_image_rs.shape[1],
                ])

                nose_vector = np.zeros((3,))

                nose_vector[0] = (detection_result.face_landmarks[0][8].x - detection_result.face_landmarks[0][2].x)
                # mediapipe returns y going down, we want it to go up (thus the minus sign in y coordinate)
                nose_vector[1] = -(detection_result.face_landmarks[0][8].y - detection_result.face_landmarks[0][2].y)
                nose_vector[2] = (detection_result.face_landmarks[0][8].z - detection_result.face_landmarks[0][2].z)
                nose_vector = nose_vector / np.linalg.norm(nose_vector)
                
                # eye vector going from right to left
                eye_vector = np.zeros((3,))
                eye_vector[0] = (detection_result.face_landmarks[0][263].x - detection_result.face_landmarks[0][33].x)
                # mediapipe returns y going down, we want it to go up (thus the minus sign in y coordinate)
                eye_vector[1] = -(detection_result.face_landmarks[0][263].y - detection_result.face_landmarks[0][33].y)
                eye_vector[2] = (detection_result.face_landmarks[0][263].z - detection_result.face_landmarks[0][33].z)
                eye_vector = eye_vector / np.linalg.norm(eye_vector)

                # we want to find the orthogonalized eye vector - Let's use Gram Schmidt process
                # so we subtract the projection of the eye vector to the nose vector from the eye vector
                # projection is <v1 . v2> * v1 (it's how much it's projeceted times v1 which is the direction
                # of the projection). This will be subtracted from the original vector to get the orthogonalized:
                # v2_orth = v2 - <v1 . v2> * v1
                # (and we suppose that both vectors are normalized, so we don't have to divide by the norms)
                eye_vector_orthogonalized = eye_vector - np.dot(eye_vector, nose_vector) * nose_vector
                # and normalize it to be sure
                eye_vector_orthogonalized = eye_vector_orthogonalized / np.linalg.norm(eye_vector_orthogonalized)

                third_orthogonal_vector = np.cross(nose_vector, eye_vector_orthogonalized)
                third_orthogonal_vector = third_orthogonal_vector / np.linalg.norm(third_orthogonal_vector)

                # this would transform points in face coordinates to the camera coordinates
                face2cam = np.column_stack((eye_vector_orthogonalized, nose_vector, third_orthogonal_vector))
                print(face2cam)

                # this would transform points in camera coordinates to the face coordinates
                # it's gonna have an inverse as it was a orthonormal matrix (it's actually its transpose..)
                cam2face = np.linalg.inv(face2cam)

        except:
            print("Mediapipe didn't detect a thing! Using the last valid values.")
            if len(forehead_points) > 0:
                forehead_points.append(forehead_points[-1])
                quality.append(0)
            else:
                duration -= 1000/30 # to account for the lost frames at the beginning

        # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = draw_face_landmarks_on_image(image.numpy_view(), detection_result)
        cv2.imshow("Face features", cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

        prev_prev_ts = prev_ts
        prev_ts = int(ts-t0)

        if prev_ts - prev_prev_ts > 2*int(1000/30) - 10:
            print(f"Skipped frame. Previous ts: {prev_ts}, difference {prev_ts - prev_prev_ts}")
            if len(forehead_points) > 0:
                forehead_points.append(forehead_points[-1])
                quality.append(0)
            else:
                raise Exception("We have to account for a shorter video duration (variable duration) because of the skipped frames at the beginning.")

        time_series.append([prev_ts])

        ch = cv2.waitKey(1)
        if ch==113: #q pressed
            break

        if len(time_series) % (60*30) == 0:
            print(f"{len(time_series) / (60*30)} minutes processed")

        t = ts - t0

        if frame_nb > 100:
            ax.cla()
            indices = np.arange(len(forehead_points))
            x = smooth(np.gradient(np.array(forehead_points)[:, 0]), 20)
            y = smooth(np.gradient(np.array(forehead_points)[:, 1]), 20)
            z = smooth(np.gradient(np.array(forehead_points)[:, 2]), 20)
            ax.plot(indices, x, label='x')
            ax.plot(indices, y, label='y')
            ax.plot(indices, z, label='z')
            clear_output(wait = True)
            plt.pause(0.00000001)

    pipeline.stop()
    cv2.destroyAllWindows()
    forehead_points = np.stack(forehead_points)
    quality = np.stack(quality)
    
    return forehead_points, quality, face2cam, cam2face, face_coordinates_origin, duration/1000

if __name__ == '__main__':

    # path_bag = "/home/mpicek/repos/master_project/test_data/corresponding/cam0_911222060374_record_13_11_2023_1330_19.bag"
    path_bag = "/home/mpicek/repos/master_project/test_data/corresponding/cam0_911222060374_record_13_11_2023_1337_20.bag"
    forehead_points, quality_data, face2cam, cam2face, face_coordinates_origin, duration = extract_face_vector(path_bag, mediapipe_face_model_file = '/home/mpicek/Downloads/face_landmarker.task')
    output_filename = 'forehead_points_20.npy'
    quality_output_filename = 'quality_20.npy'
    face2cam_output_filename = 'face2cam_20.npy'
    cam2face_output_filename = 'cam2face_20.npy'
    face_coordinates_origin_output_filename = 'face_coordinates_origin_20.npy'
    duration_output_filename = 'duration_20.npy'
    print(f"Saving data of shape {forehead_points.shape} to {output_filename}")
    print(f"Saving quality data of shape {quality_data.shape} to {quality_output_filename}")
    print(f"Saving face2cam data of shape {face2cam.shape} to {face2cam_output_filename}")
    print(f"Saving cam2face data of shape {cam2face.shape} to {cam2face_output_filename}")
    print(f"Saving face_coordinates_origin data of shape {face_coordinates_origin.shape} to {face_coordinates_origin_output_filename}")
    print(f"Saving duration data to {duration_output_filename}")
    np.save(output_filename, forehead_points)
    np.save(quality_output_filename, quality_data)
    np.save(face2cam_output_filename, face2cam)
    np.save(cam2face_output_filename, cam2face)
    np.save(face_coordinates_origin_output_filename, face_coordinates_origin)
    np.save(duration_output_filename, duration)
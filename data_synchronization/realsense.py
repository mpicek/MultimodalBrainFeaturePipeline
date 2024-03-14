import pandas as pd
import scipy.io
import pyrealsense2 as rs
import numpy as np
import cv2
import face_recognition
import matplotlib.pyplot as plt
import mediapipe as mp
from IPython.display import clear_output
from synchronization_utils import get_face_coordinate_system, get_forehead_point, recognize_patient, NoFaceDetectedException, smooth
from mediapipe_utils import *
import face_recognition


class Exception5000(Exception):
    pass

def extract_face_vector(path_bag, mediapipe_face_model_file, patients_encoding, log_table, visualize=False):
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
    cam2face, face2cam, face_coordinates_origin = None, None, None
    duration = playback.get_duration().total_seconds() * 1000
    print(f"Overall video duration: {playback.get_duration()}")

    # playback.seek(datetime.timedelta(seconds=73*4))

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
            print("SKIPPED FRAME BY THE REALSENSE!")
            if len(forehead_points) > 0:
                forehead_points.append(forehead_points[-1])
                quality.append(0)
            else:
                forehead_points.append(np.array([0, 0, 0]))
                quality.append(0)
            continue

        frame_nb = color_frame.get_frame_number()

        # end of the file (no more new frames)
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
            duration -= 1000 # for further processing we need to remember that we finished 1s earlier
            break

        if prev_ts >= int(ts-t0):
            # doubled frame or some other error in ordering (we don't include the frame
            # as we don't want it multiple times)
            continue

        if face_location is None:
            # convert it to BGR as face_recognition uses BGR

            try:
                face_location, distance = recognize_patient(color_image_rs, patients_encoding)
                top, right, bottom, left = face_location
                face_height = bottom - top
                face_width = right - left
                log_table['first_frame_face_detected'] = [color_image_rs.copy()]
                log_table['face_location'] = [face_location]
                log_table['face_patient_distance'] = [distance]
            except NoFaceDetectedException:
                forehead_points.append(np.array([0, 0, 0]))
                quality.append(0)
                continue

        # Crop the face
        # we cut the face with some margin around it (defined by around_face_factor * face_width or face_height)
        # but we have to be careful not to go out of the image (that's why all the max and min functions)                    
        face_image = color_image_rs[
            max(0, top - int(face_height * around_face_factor)):min(img_height, int(bottom + face_height * around_face_factor)),
            max(0, left - int(face_width * around_face_factor)):min(img_width, int(right + face_width * around_face_factor))
        ]

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        detection_result = detector.detect(image)
        
        # getting the forehead points
        try:
            
            forehead_points.append(get_forehead_point(detection_result, face_image))
            quality.append(1)

            if cam2face is None:
                # We find the face coordinate system in the first detected frame (this will be the reference)
                # It's gonna be in matrix [x, y, z] where x, y, z are the bases of the coordinate system
                face_coordinates_origin, face2cam, cam2face = get_face_coordinate_system(detection_result, color_image_rs)
                print(cam2face)

        except:
            print("Mediapipe didn't detect a thing! Using the last valid values.")
            if len(forehead_points) > 0:
                forehead_points.append(forehead_points[-1])
                quality.append(0)
            else:
                forehead_points.append(np.array([0, 0, 0]))
                quality.append(0)


        # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = draw_face_landmarks_on_image(image.numpy_view(), detection_result)
        if visualize:
            cv2.imshow("Face features", cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

        prev_prev_ts = prev_ts
        prev_ts = int(ts-t0)

        if prev_ts - prev_prev_ts > 2*int(1000/30) - 10:
            print(f"Skipped frame. Previous ts: {prev_ts}, difference {prev_ts - prev_prev_ts}")
            if len(forehead_points) > 0:
                forehead_points.append(forehead_points[-1])
                quality.append(0)
            else:
                forehead_points.append(np.array([0, 0, 0]))
                quality.append(0)
                raise Exception("We have to account for a shorter video duration (variable duration) because of the skipped frames at the beginning.")

        time_series.append([prev_ts])

        ch = cv2.waitKey(1)
        if ch==113: #q pressed
            break

        if len(time_series) % (60*30) == 0:
            print(f"{len(time_series) / (60*30)} minutes processed")

        t = ts - t0

        if frame_nb > 100 and visualize:
            ax.cla()
            indices = np.arange(len(forehead_points))
            x = smooth(np.gradient(np.array(forehead_points)[:, 0]), 20)
            y = smooth(np.gradient(np.array(forehead_points)[:, 1]), 20)
            z = smooth(np.gradient(np.array(forehead_points)[:, 2]), 20)
            # add name of the plot
            ax.set_title('First derivative of the forehead points (=velocity)')
            ax.plot(indices, x, label='x')
            ax.plot(indices, y, label='y')
            ax.plot(indices, z, label='z')
            clear_output(wait = True)
            plt.pause(0.00000001)

    pipeline.stop()
    cv2.destroyAllWindows()
    # change all [0, 0, 0] at the beginning to the first valid value (these [0, 0, 0] are the
    # consequence of errors at the beginning or not visible face). I change it to the next valid value
    # so that there is not that big jump in the positions (in this way it means the face is not moving,
    # but quality is still 0 so it will not be accounted for in the cross-correlation during the synchronization)
    forehead_points = np.stack(forehead_points)
    non_zero_index = np.where((forehead_points != [0, 0, 0]).any(axis=1))[0][0]
    forehead_points[:non_zero_index] = forehead_points[non_zero_index]

    quality = np.stack(quality)
    log_table['avg_quality'] = [quality.mean()]

    
    return forehead_points, quality, face2cam, cam2face, face_coordinates_origin, duration/1000, log_table

if __name__ == '__main__':

    # path_bag = "/home/mpicek/repos/master_project/test_data/corresponding/cam0_911222060374_record_13_11_2023_1330_19.bag"
    path_bag = "/home/mpicek/repos/master_project/test_data/corresponding/cam0_911222060374_record_13_11_2023_1337_20.bag"

    patient_img = face_recognition.load_image_file('/home/mpicek/repos/master_project/data_synchronization/patient.png')
    patients_encoding = face_recognition.face_encodings(patient_img)[0] # warning, it's a list!! That's why [0]

    forehead_points, quality_data, face2cam, cam2face, face_coordinates_origin, duration = extract_face_vector(path_bag, mediapipe_face_model_file='/home/mpicek/Downloads/face_landmarker.task', patients_encoding=patients_encoding, visualize=True)
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
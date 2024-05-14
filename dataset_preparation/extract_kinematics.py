from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import matplotlib.pyplot as plt
import av
from fractions import Fraction
import os
import argparse
from tqdm import tqdm
import multiprocessing


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def process_video_with_occlusions(video_path, detector, occlusions_path, annotated_video_name=None):

    occlusions = np.load(occlusions_path)

    if annotated_video_name is not None:
        container = av.open(annotated_video_name, mode='w')
        stream = container.add_stream('mpeg4')

    cap = cv2.VideoCapture(video_path)
    frame_nb = 0
    joints = []

    in_dataset = []

    # get the number of frames
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # get video resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if annotated_video_name is not None:
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        stream.framerate = 30

    # while cap.isOpened():
    # use tqdm to show progress bar

    for _ in tqdm(range(n_frames)):
        if not cap.isOpened():
            break

        success, frame = cap.read()

        if success == True:

            # If there is an occlusion (occlusion == 1) or the patient was not detected at all (occlusion == 2)
            if occlusions[frame_nb] != 0:
                frame_nb += 1
                joints.append([[np.nan, np.nan, np.nan, np.nan] for _ in range(33)])
                in_dataset.append(0)
                if annotated_video_name is not None:
                    frame_saved = av.VideoFrame.from_ndarray(frame)
                    for packet in stream.encode(frame_saved):
                        container.mux(packet)
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = detector.detect_for_video(mp_img, int((frame_nb / 30) * 1000)) # timestamp in ms
            result = detection_result.pose_landmarks
            if result == []:
                in_dataset.append(0)
                joints.append([[np.nan, np.nan, np.nan, np.nan] for _ in range(33)])
            else:
                in_dataset.append(1)
                joints.append([[landmark.x * width, landmark.y * height, landmark.z * width, landmark.visibility] for landmark in result[0]])
            annotated_image = draw_landmarks_on_image(mp_img.numpy_view(), detection_result)

            frame = av.VideoFrame.from_ndarray(annotated_image)

            if annotated_video_name is not None:
                for packet in stream.encode(frame):
                    container.mux(packet)

        else:
            break
        frame_nb += 1

    if annotated_video_name is not None:
        for packet in stream.encode():
            container.mux(packet)
        container.close()

    cap.release()
    return np.array(joints), np.array(in_dataset)

def wrapper(mp4_basename, mp4_folder, occlusions_folder, output_folder, mediapipe_model_path):
    base_options = python.BaseOptions(model_asset_path=mediapipe_model_path)
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = vision.PoseLandmarkerOptions(
        running_mode=VisionRunningMode.VIDEO,
        base_options=base_options,
        output_segmentation_masks=True,
        num_poses=5,
        min_pose_detection_confidence=0.8,
        min_tracking_confidence=0.7
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    print(f"Processing file {mp4_basename}")
    path_mp4 = os.path.join(mp4_folder, mp4_basename)
    path_occlusions = os.path.join(occlusions_folder, mp4_basename[:-len("_cropped.mp4")] + '_occlusion.npy')
    path_output = os.path.join(output_folder, mp4_basename[:-len("_cropped.mp4")] + '_kinematics.npy')
    path_annotated_video = os.path.join(output_folder, mp4_basename[:-len("_cropped.mp4")] + '_annotated.mp4')

    if os.path.exists(path_output):
        print(f"\tFile already processed. Skipping.")
        return
    
    if not os.path.exists(path_occlusions):
        print(f"\tNo occlusions file found. Skipping.")
        return

    joints, in_dataset = process_video_with_occlusions(path_mp4, detector, path_occlusions, path_annotated_video)
    np.save(path_output, joints)
    np.save(path_output[:-len("_kinematics.npy")] + '_in_dataset.npy', in_dataset)


def main(mp4_folder, occlusions_folder, output_folder, mediapipe_model_path):
    
    mp4_basenames = [file for file in os.listdir(mp4_folder) if file.endswith('.mp4')]

    for mp4_basename in mp4_basenames:
        wrapper(mp4_basename, mp4_folder, occlusions_folder, output_folder, mediapipe_model_path)

    # pool = multiprocessing.Pool()

    # pool.starmap(wrapper, [(mp4_basename, mp4_folder, occlusions_folder, output_folder, mediapipe_model_path) for mp4_basename in mp4_basenames])
    # pool.close()
    # pool.join()
    

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Extract the LED signals from .mp4 files.")
    parser.add_argument("cropped_mp4_videos_folder", help="Path to the folder containing the cropped mp4 files.")
    parser.add_argument("occlusions_folder", help="Path to the folder containing occlusions files.")
    parser.add_argument("output_folder", help="Where to put output.")
    parser.add_argument("--mediapipe_model_path", default='/home/vita-w11/mpicek/master_project/mediapipe_models/pose_landmarker_heavy.task', help="Path to the mediapipe model.")    

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    main(args.cropped_mp4_videos_folder, args.occlusions_folder, args.output_folder, args.mediapipe_model_path)
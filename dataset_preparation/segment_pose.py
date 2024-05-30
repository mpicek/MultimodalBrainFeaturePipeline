import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import av
import os
import argparse
from tqdm import tqdm

def process_video_segment(video_path, detector, occlusions_path, output_video_name, delete_occlusions):
    occlusions = np.load(occlusions_path)
    
    cap = cv2.VideoCapture(video_path)
    frame_nb = 0

    container = av.open(output_video_name, mode='w')
    stream = container.add_stream('mpeg4')

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'
    stream.framerate = 30

    green_background = np.full((height, width, 3), (0, 255, 0), dtype=np.uint8)

    for _ in tqdm(range(n_frames)):
        if not cap.isOpened():
            break

        success, frame = cap.read()
        if not success:
            break

        if occlusions[frame_nb] == 1 and delete_occlusions:
            frame_saved = av.VideoFrame.from_ndarray(green_background)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = detector.detect_for_video(mp_img, int((frame_nb / 30) * 1000))

            if occlusions[frame_nb] == 1:
                frame_saved = av.VideoFrame.from_ndarray(green_background)
            else:
                if detection_result.segmentation_masks:
                    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
                    mask_3d = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2)
                    # mask_3d = mask_3d.astype(np.uint8)
                    masked_frame = np.where(mask_3d > MASK_THRESHOLD, frame, green_background)
                    frame_saved = av.VideoFrame.from_ndarray(masked_frame)
                else:
                    frame_saved = av.VideoFrame.from_ndarray(green_background)

        for packet in stream.encode(frame_saved):
            container.mux(packet)

        frame_nb += 1

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    cap.release()

def wrapper(mp4_basename, mp4_folder, occlusions_folder, output_folder, mediapipe_model_path, delete_occlusions):
    base_options = python.BaseOptions(model_asset_path=mediapipe_model_path)
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = vision.PoseLandmarkerOptions(
        running_mode=VisionRunningMode.VIDEO,
        base_options=base_options,
        output_segmentation_masks=True,
        num_poses=1,
        min_pose_detection_confidence=0.8,
        min_tracking_confidence=0.7
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    print(f"Processing file {mp4_basename}")
    path_mp4 = os.path.join(mp4_folder, mp4_basename)
    path_occlusions = os.path.join(occlusions_folder, mp4_basename[:-len("_cropped.mp4")] + '_occlusion.npy')
    path_output_video = os.path.join(output_folder, mp4_basename[:-len("_cropped.mp4")] + '_segmented_pose.mp4')
    temp_output_video = path_output_video.replace('.mp4', '_temp.mp4')

    if os.path.exists(temp_output_video):
        os.remove(temp_output_video)

    if os.path.exists(path_output_video):
        print(f"\tFile already processed. Skipping.")
        return

    if not os.path.exists(path_occlusions):
        print(f"\tNo occlusions file found. Skipping.")
        return

    process_video_segment(path_mp4, detector, path_occlusions, temp_output_video, delete_occlusions)
    os.rename(temp_output_video, path_output_video)

    if os.path.exists(temp_output_video):
        os.remove(temp_output_video)

def main(mp4_folder, occlusions_folder, output_folder, mediapipe_model_path, delete_occlusions):
    mp4_basenames = [file for file in os.listdir(mp4_folder) if file.endswith('.mp4')]

    for i, mp4_basename in enumerate(mp4_basenames):
        
        print(f"Processing {str(i)}/{len(mp4_basenames)}:", mp4_basename)

        wrapper(mp4_basename, mp4_folder, occlusions_folder, output_folder, mediapipe_model_path, delete_occlusions)

MASK_THRESHOLD = 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment the pose from .mp4 files.")
    parser.add_argument("cropped_mp4_videos_folder", help="Path to the folder containing the cropped mp4 files.")
    parser.add_argument("occlusions_folder", help="Path to the folder containing occlusions files.")
    parser.add_argument("output_folder", help="Where to put output videos.")
    parser.add_argument("--mediapipe_model_path", default='../pretrained_models/pose_landmarker_heavy.task', help="Path to the mediapipe model.")
    parser.add_argument("--delete_occlusions", action='store_true', help="Delete occluded frames.")

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    main(args.cropped_mp4_videos_folder, args.occlusions_folder, args.output_folder, args.mediapipe_model_path, args.delete_occlusions)

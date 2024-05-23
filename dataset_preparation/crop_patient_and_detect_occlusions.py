###############################################################################################
#
# Logs only files that were skipped. We skip a file in the following cases:
#   - the nose position likelihood is too low (so the patient is not there or he/she is backwards or DLC just didn't detect it) - when likelihood is < 0.4
#   - no person is at the location of the detected patient's nose
#   - no people detected in the video
# If the patient is not found, the bounding box is created with just None value.
# Finds patient in the video based on the detection with YOLO and the nose position from DLC
# Skips the processing of files that have already been processed (based on the existence of the _bounding_box.npy file or presence in skipped_files.csv)
# Saves the bounding box of the patient and the cropped image (for visualization)
# Then detects occlusions (when the patient's bounding box is overlapping with another person's bounding box) 
# CAREFUL: THE BOUNDING_BOXES MIGHT CONSIST JUST FROM THE NONE VALUE WHEN NO PERSON IS DETECTED
# OCCLUSION FILE IS MADE ONLY WHEN EVERYTHING IS SUCCESSFUL. SO WHEN THE FILE IS MISSING, IT MEANS THAT THE FILE WAS SKIPPED.
#
###############################################################################################

import sys
sys.path.append('../data_synchronization')
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2
import os
import argparse
from synchronization_utils import extract_movement_from_dlc_csv
from tqdm import tqdm

class NoPatientDetectedError(Exception):
    pass

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def process_video(video_path, model, analyze_every_n_frame=15):

    total_frames = count_frames(video_path)
    progress_bar = tqdm(total=total_frames / analyze_every_n_frame, desc="Processing frames", unit="frame")

    cap = cv2.VideoCapture(video_path)

    # bounding boxes (and corresponding patient ids) for each frame
    bounding_boxes, patient_ids = [], []
    video_height, video_width = None, None
    example_frame = None

    frame_nb = -1

    while cap.isOpened():
        success, frame = cap.read()

        if success == True:
            frame_nb += 1
            if frame_nb % analyze_every_n_frame == 0:
                progress_bar.update(1)
                example_frame = frame
                video_height = frame.shape[0]
                video_width = frame.shape[1]

                # persist .. to take into account temporal information from the previous frames
                # classes=[0] .. to track only the person class
                results = model.track(frame, persist=True, classes=[0], verbose=False)

                try:
                    boxes = results[0].boxes.xyxy.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    bounding_boxes.append(boxes)
                    patient_ids.append(track_ids)
                except:
                    bounding_boxes.append(None)
                    patient_ids.append(None)
        else:
            break

    progress_bar.close()
    cap.release()

    return bounding_boxes, patient_ids, video_height, video_width, example_frame, frame_nb + 1

def get_nose_position_from_DLC(dlc_path):

    movement, likelihood = extract_movement_from_dlc_csv(dlc_path)
    avg_x, avg_y = movement.mean(axis=0)
    if np.mean(likelihood) < 0.4:
        raise NoPatientDetectedError(f"The average nose position likelihood is too low ({np.mean(likelihood.mean())}).")
    return avg_x, avg_y

def get_patients_bounding_box(bounding_boxes, patient_ids, video_height, video_width, nose_x, nose_y):

    ##################################
    # find unique ids (=unique people)
    id_list = []
    for ids in patient_ids:
        id_list.extend(ids)
    unique_ids = list(set(id_list))

    ##################################
    # compute the average bounding box for each person
    peoples_avg_bbs = {}

    # go over each person and extract their positions
    for id in unique_ids:
        stacked_positions = []
        for frame in range(len(patient_ids)):
            if id in patient_ids[frame]: # if the person is in the frame
                idx = patient_ids[frame].index(id)
                stacked_positions.append(bounding_boxes[frame][idx])
        stacked_positions = np.array(stacked_positions)

        averaged_position = np.mean(stacked_positions, axis=0)
        peoples_avg_bbs[id] = averaged_position

    if not peoples_avg_bbs:
        raise NoPatientDetectedError("No people detected in the video.")

    ##################################
    # find the patient's id based on the nose position (it has to be inside the bounding box)
    patient_id = None
    patient_height, patient_width = None, None

    for id, pos in peoples_avg_bbs.items():
        if pos[0] < nose_x and nose_x < pos[2] and pos[1] < nose_y and nose_y < pos[3]:
            # remember the larger patient if there are multiple patients
            # (this is because it's better to have more surrounding area around the patient than less,
            #  we don't want to lose information)
            if patient_height is None or (pos[3] - pos[1]) > patient_height:
                patient_id = id
                patient_height = pos[3] - pos[1]
                patient_width = pos[2] - pos[0]
    
    if patient_id is None:
        raise NoPatientDetectedError("No person is at the location of the detected patient's nose.")

    patient_bb = peoples_avg_bbs[patient_id].copy()
    patient_bb_small = peoples_avg_bbs[patient_id].copy()
    
    patient_bb[0] = int(max(0, patient_bb[0] - COEF_WIDTH * patient_width))
    patient_bb[1] = int(max(0, patient_bb[1] - COEF_HEIGHT * patient_height))
    patient_bb[2] = int(min(video_width, patient_bb[2] + COEF_WIDTH * patient_width))
    patient_bb[3] = int(min(video_height, patient_bb[3] + COEF_HEIGHT * patient_height))

    return patient_bb, patient_bb_small, patient_id



def main(mp4_folder, dlc_folder, extraction_folder, DLC_suffix, analyzed_seconds, fps):
    model = YOLO('yolov8n.pt')
    analyze_every_n_frame = 15
    
    for mp4_name in os.listdir(mp4_folder):
        if not mp4_name.endswith('.mp4'):
            continue

        mp4_basename = mp4_name[:-4]
        if os.path.exists(os.path.join(extraction_folder, mp4_basename + '_bounding_box.npy')):
            print(f"File {mp4_name} already processed. Skipping")
            continue

        # if skipped_files.csv file exists, and the file is in it, skip it
        if os.path.exists(os.path.join(extraction_folder, 'skipped_files.csv')):
            with open(os.path.join(extraction_folder, 'skipped_files.csv'), 'r') as f:
                skipped_files = f.read().splitlines()
            if mp4_name in skipped_files:
                print(f"File {mp4_name} is in the skipped files list. Skipping.")
                continue

        print("Processing file:", mp4_name)
        path_mp4 = os.path.join(mp4_folder, mp4_name)
        bounding_boxes, patient_ids, video_height, video_width, example_frame, num_frames = process_video(path_mp4, model, analyze_every_n_frame=analyze_every_n_frame)

        # remove Nones from bounding boxes and patient ids
        no_None_bounding_boxes = [bb for bb in bounding_boxes if bb is not None]
        no_None_patient_ids = [ids for ids in patient_ids if ids is not None]

        dlc_csv_path = os.path.join(dlc_folder, mp4_basename + DLC_suffix)
        try:
            average_nose_x, average_nose_y = get_nose_position_from_DLC(dlc_csv_path)
        except NoPatientDetectedError as e:
            print(str(e))
            print(f"Could not find nose position in {dlc_csv_path}. Skipping.")
            # print skipped files into a log file
            with open(os.path.join(extraction_folder, 'skipped_files.csv'), 'a') as f:
                f.write(f"{mp4_name}\n")
            continue
        try:
            bounding_box, patient_bb_small, patient_id = get_patients_bounding_box(no_None_bounding_boxes, no_None_patient_ids, video_height, video_width, average_nose_x, average_nose_y)
            cropped = example_frame[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]
            bounding_box = bounding_box.astype(int)
            np.save(os.path.join(extraction_folder, mp4_basename + '_bounding_box.npy'), bounding_box)
            cv2.imwrite(os.path.join(extraction_folder, mp4_basename + '_cropped.png'), cropped)

            ################### DETECT OCCLUSIONS ###################
            occlusion = []
            # go over all frames and set occlusion to 1 when the patient_id's bounding box overlaps with another patient
            for f in range(len(bounding_boxes)):
                occlusion.append(0)
                if bounding_boxes[f] is None:
                    occlusion[f] = 1  # nobody is in the frame so we mark it as occlusion (we don't want to analyze the frame later)
                    continue
                occlusion_count = 0
                for i in range(len(bounding_boxes[f])):
                    if patient_ids[f][i] != patient_id:
                        left1, top1, right1, bottom1 = patient_bb_small
                        left2, top2, right2, bottom2 = bounding_boxes[f][i]

                        if (left1 < right2) and (right1 > left2) and (top1 < bottom2) and (bottom1 > top2):
                            occlusion_count += 1
                            overlap_width = min(right1, right2) - max(left1, left2)
                            patient_width = right1 - left1

                            if occlusion_count > 1:
                                occlusion[f] = 1  # full occlusion due to multiple overlaps
                                break

                            if overlap_width > 0.25 * patient_width:
                                occlusion[f] = 1  # full occlusion
                                break
                            else:
                                if left1 < left2 and right1 > left2:
                                    occlusion[f] = 2  # occlusion from the right - but from the camera view, so it is on the patient's left
                                elif left1 < right2 and right1 > right2:
                                    occlusion[f] = 3  # occlusion from the left - but from the camera view, so it is on the patient's right


            # expand the occlusions to be for every frame and not for just every analyzed frame
            occlusion = np.repeat(occlusion, analyze_every_n_frame)
            occlusion = occlusion[:num_frames]

            np.save(os.path.join(extraction_folder, mp4_basename + '_occlusion.npy'), occlusion)

        except NoPatientDetectedError as e:
            print(str(e))
            np.save(os.path.join(extraction_folder, mp4_basename + '_bounding_box.npy'), np.array(None))
            with open(os.path.join(extraction_folder, 'skipped_files.csv'), 'a') as f:
                f.write(f"{mp4_name}\n")

COEF_WIDTH = 0.3
COEF_HEIGHT = 0.2

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Crop the patient from .mp4 files.")
    parser.add_argument("mp4_folder", help="Path to the folder containing mp4 files.")
    parser.add_argument("dlc_folder", help="Path to the folder DLC files.")
    parser.add_argument("extraction_folder", help="Path to save the extraction.")
    args = parser.parse_args()

    if not os.path.exists(args.extraction_folder):
        os.makedirs(args.extraction_folder)

    analyzed_seconds = 13
    fps = 30
    DLC_suffix = 'DLC_resnet50_UP2_movement_syncApr15shuffle1_525000.csv'
    main(args.mp4_folder, args.dlc_folder, args.extraction_folder, DLC_suffix, analyzed_seconds, fps)
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

def process_video(video_path, model, analyzed_seconds=13, fps=30):
    cap = cv2.VideoCapture(video_path)

    frame_nb = 1
    fps = 30

    # bounding boxes (and corresponding patient ids) for each frame
    bounding_boxes, patient_ids = [], []
    video_height, video_width = None, None
    example_frame = None

    while cap.isOpened():
        success, frame = cap.read()

        if success == True:
            # if frame_nb % 30 == 0:
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
                pass
        else:
            break
        # frame_nb += 1
        # if frame_nb > analyzed_seconds * fps:
        #     break
    cap.release()

    return bounding_boxes, patient_ids, video_height, video_width, example_frame

def get_nose_position_from_DLC(dlc_path):

    movement, _ = extract_movement_from_dlc_csv(dlc_path)
    avg_x, avg_y = movement.mean(axis=0)
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
    coef_width = 0.3
    coef_height = 0.2
    patient_bb[0] = int(max(0, patient_bb[0] - coef_width * patient_width))
    patient_bb[1] = int(max(0, patient_bb[1] - coef_height * patient_height))
    patient_bb[2] = int(min(video_width, patient_bb[2] + coef_width * patient_width))
    patient_bb[3] = int(min(video_height, patient_bb[3] + coef_height * patient_height))

    return patient_bb, patient_bb_small, patient_id



def main(mp4_folder, dlc_folder, extraction_folder, log_table, DLC_suffix, analyzed_seconds, fps):
    model = YOLO('yolov8n.pt')
    
    log_table = pd.read_csv(log_table)

    # if column "patient_bb" is not in the log table, add it
    if 'patient_bb' not in log_table.columns:
        log_table['patient_bb'] = None
    
    # go over mp4_name in log table that have sync_failed == 0
    for mp4_name in tqdm(log_table.loc[log_table['sync_failed'] == 0, 'mp4_name']):
        mp4_basename = mp4_name[:-4]
        if os.path.exists(os.path.join(extraction_folder, mp4_basename + '_bounding_box.npy')):
            print(f"File {mp4_name} already processed. Skipping")
            continue
        path_mp4 = os.path.join(mp4_folder, mp4_name)
        bounding_boxes, patient_ids, video_height, video_width, example_frame = process_video(path_mp4, model, analyzed_seconds, fps)

        dlc_csv_path = os.path.join(dlc_folder, mp4_basename + DLC_suffix)
        average_nose_x, average_nose_y = get_nose_position_from_DLC(dlc_csv_path)
        try:
            bounding_box, patient_bb_small, patient_id = get_patients_bounding_box(bounding_boxes, patient_ids, video_height, video_width, average_nose_x, average_nose_y)
            cropped = example_frame[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]
            bounding_box = bounding_box.astype(int)
            np.save(os.path.join(extraction_folder, mp4_basename + '_bounding_box.npy'), bounding_box)
            cv2.imwrite(os.path.join(extraction_folder, mp4_basename + '_cropped.png'), cropped)

            ################### DETECT OCCLUSIONS ###################
            occlusion = []
            # go over all frames and set occlusion to 1 when the patient_id's bounding box overlaps with another patient
            for f in range(len(bounding_boxes)):
                occlusion.append(0)
                for i in range(len(bounding_boxes[f])):
                    patient_index = patient_ids[f].index(patient_id)
                    if i != patient_index:
                        left1, top1, right1, bottom1 = bounding_boxes[f][patient_index]
                        left2, top2, right2, bottom2 = bounding_boxes[f][i]

                        if (left1 < right2) and (right1 > left2) and (top1 < bottom2) and (bottom1 > top2):
                            occlusion[f] = 1
                            break
        except NoPatientDetectedError as e:
            continue


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Crop the patient from .mp4 files.")
    parser.add_argument("mp4_folder", help="Path to the folder containing mp4 files.")
    parser.add_argument("dlc_folder", help="Path to the folder DLC files.")
    parser.add_argument("extraction_folder", help="Path to save the extraction.")
    parser.add_argument("log_table", help="Path to the log table.")
    args = parser.parse_args()

    if not os.path.exists(args.extraction_folder):
        os.makedirs(args.extraction_folder)

    analyzed_seconds = 13
    fps = 30
    DLC_suffix = 'DLC_resnet50_UP2_movement_syncApr15shuffle1_525000.csv'
    main(args.mp4_folder, args.dlc_folder, args.extraction_folder, args.log_table, DLC_suffix, analyzed_seconds, fps)
import argparse

from FaceMovementExtractor import FaceMovementExtractor
from Synchronizer import Synchronizer
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process directories for Realsense and Wisci data.")
    parser.add_argument("--realsense_server_path", type=str, required=True, help="Path to the Realsense server directory")
    parser.add_argument("--wisci_server_path", type=str, required=True, help="Path to the Wisci server directory")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder")

    args = parser.parse_args()

    face_movement_extractor = FaceMovementExtractor(
        realsense_server_path=args.realsense_server_path, 
        output_folder=args.output_folder, 
        mediapipe_face_model_file='/home/mpicek/Downloads/face_landmarker.task', 
        patient_image_path="/home/mpicek/repos/master_project/data_synchronization/patient.png", 
        log_table_path=os.path.join(args.output_folder, 'table.csv'), 
        visualize=True
    )

    face_movement_extractor.process_directories()

    synchronizer = Synchronizer(
        wisci_server_path=args.wisci_server_path, 
        output_folder=args.output_folder, 
        log_table_path=os.path.join(args.output_folder, 'table.csv'),
        box_size=40,
        visualize=False
    )

    synchronizer.process_all_realsense_recordings()
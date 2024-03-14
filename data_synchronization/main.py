import numpy as np
import pandas as pd
import argparse
import os
import re
import face_recognition
from realsense import extract_face_vector, Exception5000
from synchronization_utils import log_table_columns

class Synchronizer:
    def __init__(self, wisci_server_path, realsense_server_path, output_folder, mediapipe_face_model_file, patient_image_path, log_table_path, synchronize, visualize=False):
        self.mediapipe_face_model_file = mediapipe_face_model_file
        patient_img = face_recognition.load_image_file(patient_image_path)
        self.patients_encoding = face_recognition.face_encodings(patient_img)[0] # warning, it's a list!! That's why [0]

        self.wisci_server_path = wisci_server_path
        self.realsense_server_path = realsense_server_path
        self.output_folder = output_folder
        self.log_table_path = log_table_path
        self.synchronize = synchronize
        self.visualize = visualize

    def process_directories(self):
        """
        Supposing the .bag files are in the deepest subdirectories of the realsense_server_path,

        realsense_server_path
        ├── 2021_01_01_session
        │   ├── realsense_right
        │   │   ├── 2021_01_01_01_01.bag
        │   │   ├── 2021_01_01_01_01.something
        |   ├── realsense_L
        │   │   ├── 2021_01_01_01_01.bag
        ├── 2021_01_02_session_realsense_files
        │   ├── 2021_01_01_01_01.bag
        │   ├── 2021_01_01_01_01.something
        │   ├── 2021_01_01_01_02.bag

        """
        # generates all possible routes that exist in the realsense_server_path
        for root, _, files in os.walk(self.realsense_server_path):
            relpath = os.path.relpath(root, self.realsense_server_path)

            output_subdir = os.path.join(self.output_folder, relpath)
            date = self._realsense_day_extract(relpath)
            if os.path.exists(output_subdir):
                print(f"The subdirectory {output_subdir} exists, skipping.")
                continue

            for file in files:
                print(f"Processing {file}")
                if file.endswith(".bag"):
                    os.makedirs(output_subdir, exist_ok=True)
                    
                    # full path to the output folder named by the filename but without the .bag extension
                    output_file_path = os.path.join(output_subdir, file[:-4])

                    path_bag = os.path.join(root, file) # full path to the file
                    print(f"Processing {path_bag} and saving to {output_file_path}")

                    log_table = self.process_bag_and_save(path_bag, output_file_path)
                    log_table['date'] = [date]

                    if self.synchronize and log_table['failed'][0] == 0:
                        possible_mat_files = self.find_relevant_mat_files(log_table)
                        log_table['relevant_wisci_files'] = [possible_mat_files]

                        self._append_row_and_save(log_table)
                    else:
                        self._append_row_and_save(log_table)

    
    def _realsense_day_extract(self, folder_name):
        """
        Extracts the date from a Realsense folder name, searching anywhere in the string.

        Parameters:
        folder_name (str): The name of the folder, which may contain a date in the format Year_month_day anywhere.

        Returns:
        str: The extracted date in the format "Year_month_day", or None if no date is found.
        """
        # Updated regex to search for the date pattern anywhere in the string
        match = re.search(r"(\d{4}_\d{2}_\d{2})", folder_name)
        if match:
            return match.group(1)
        else:
            return None

    def _append_row_and_save(self, new_row_df):
        """
        Appends the log table to the old saved log table (the current new_row_df has just one row,
        so it appends it to the old log table and saves it). If the old one doesn't exist, it creates a new one.

        Parameters:
        - new_row_df (pandas.df): pandas df with just one row
        """
        
        # Check if the CSV file exists
        if os.path.exists(self.log_table_path):
            # If the file exists, load the existing DataFrame, but only the header to get column names
            header_df = pd.read_csv(self.log_table_path, nrows=0)
            
            # Ensure the new row has the same columns as the existing CSV, in the same order
            # This step is crucial to avoid misalignment issues
            new_row_df = new_row_df.reindex(columns=header_df.columns)
            
            # Append the new row DataFrame to the CSV file without writing the header again
            new_row_df.to_csv(self.log_table_path, mode='a', header=False, index=False)
        else:
            # If the file does not exist, save the new DataFrame as the CSV file with a header
            new_row_df.to_csv(self.log_table_path, index=False)


    def process_bag_and_save(self, path_bag, output_file_path):
        """
        Processes a bag file to extract forehead points, saves the outputs, and logs the process.

        Parameters:
        - path_bag (str): Path to the bag file to be processed.
        - output_file_path (str): Directory path where output files will be saved.
        - visualize (bool, optional): If True, visualizes the process. Defaults to False.

        Returns:
        - log_table (DataFrame): A pandas DataFrame containing the log of the operation, including paths, success/failure, and any errors encountered.

        The function attempts to extract facial vectors from the specified bag file using the provided MediaPipe face model and patient encodings. It logs the process, including whether it succeeded or failed, and saves the extracted data to the specified output path. In case of exceptions, it logs the error and marks the operation as failed.
        """

        log_table = pd.DataFrame(columns=log_table_columns)
        log_table['path_bag'] = [path_bag]
        log_table['output_path'] = [output_file_path]

        try:
            forehead_points, quality_data, face2cam, cam2face, face_coordinates_origin, duration, log_table = extract_face_vector(path_bag, mediapipe_face_model_file=self.mediapipe_face_model_file, patients_encoding=self.patients_encoding, log_table=log_table, visualize=self.visualize)
            log_table['failed'] = [0]
            os.makedirs(output_file_path, exist_ok=True)
            np.save(os.path.join(output_file_path, 'forehead_points.npy'), forehead_points)
            np.save(os.path.join(output_file_path, 'quality.npy'), quality_data)
            np.save(os.path.join(output_file_path, 'face2cam.npy'), face2cam)
            np.save(os.path.join(output_file_path, 'cam2face.npy'), cam2face)
            np.save(os.path.join(output_file_path, 'face_coordinates_origin.npy'), face_coordinates_origin)
            np.save(os.path.join(output_file_path, 'duration.npy'), duration)
        except Exception5000:
            log_table['Exception5000'] = [1]
            log_table['failed'] = [1]
        except Exception as e:
            log_table['unknown_error_extraction'] = [str(e)]
            log_table['failed'] = [1]

        return log_table

    def find_relevant_mat_files(self, log_table):
        """
        wisci_server_path
        ├── 2021_01_01_session
        │   ├── wisci
        │   │   ├── 2021_01_01_01_01.mat
        │   │   ├── 2021_01_01_01_02.mat
        │   │   ├── 2021_01_01_01_03.mat
        │   │   ├── 2021_01_01_01_03.something
        ├── 2021_01_02_session
        │   ├── wimagine
        │   │   ├── 2021_01_01_01_01.mat
        │   │   ├── 2021_01_01_01_01.something
        """
        date = log_table['date'][0]
        print(date)
        possible_mat_files = []
        for folder in os.listdir(self.wisci_server_path):
            print(folder)
            if date not in folder:
                continue
            print(f"\tFound folder {folder} for date {date}")

            day_folder = os.path.join(self.wisci_server_path, folder)
            # now find wisci folder
            for day_subfolder in os.listdir(day_folder):
                # if includes wisci (case insensitive)
                if 'wisci' not in day_subfolder.lower() and 'wimagine' not in day_subfolder.lower():
                    continue

                print(f"Found wisci folder {day_subfolder} for date {date}")

                wisci_folder = os.path.join(day_folder, day_subfolder)

                # now in wisci folder, there is many subfolders, each containing just one .mat file,
                # let's find those .mat file
                for root, _, files in os.walk(wisci_folder):
                    for file in files:
                        if file.endswith('.mat'):
                            mat_file = os.path.join(root, file)
                            possible_mat_files.append(mat_file)
        
        return possible_mat_files    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process directories for Realsense and Wisci data.")
    parser.add_argument("--realsense_server_path", type=str, required=True, help="Path to the Realsense server directory")
    parser.add_argument("--wisci_server_path", type=str, required=True, help="Path to the Wisci server directory")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder")

    args = parser.parse_args()

    # print(args)
    synchronizer = Synchronizer(
        wisci_server_path=args.wisci_server_path, 
        realsense_server_path=args.realsense_server_path, 
        output_folder=args.output_folder, 
        mediapipe_face_model_file='/home/mpicek/Downloads/face_landmarker.task', 
        patient_image_path="/home/mpicek/repos/master_project/data_synchronization/patient.png", 
        log_table_path="/home/mpicek/repos/master_project/processed2/table.csv", 
        synchronize=True, 
        visualize=True
    )

    synchronizer.process_directories()
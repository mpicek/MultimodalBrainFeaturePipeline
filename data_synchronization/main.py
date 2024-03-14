import numpy as np
import pandas as pd
import argparse
import os
import re
import face_recognition
from realsense import extract_face_vector, Exception5000
from synchronization_utils import log_table_columns

class FaceMovementExtractor:
    """
    Extracts face movements from RealSense video.

    Attributes:
        mediapipe_face_model_file (str): Path to the MediaPipe face model file.
        patients_encoding (numpy.ndarray): Encoded face data of the patient for recognition.
        realsense_server_path (str): Base directory path where Realsense data is stored.
        output_folder (str): Directory path where processed outputs will be saved.
        log_table_path (str): Path to the CSV file used for logging processing details.
        visualize (bool): Flag to enable or disable visualization of the processing. Defaults to False.

    Methods:
        process_directories(): Processes directories containing .bag files, extracts relevant data, and logs the process.
        _realsense_day_extract(folder_name): Extracts the date from a Realsense folder name.
        _append_row_and_save(new_row_df): Appends a new row to the log table and saves it.
        process_bag_and_save(path_bag, output_file_path): Processes a .bag file, saves extracted data, and logs the process.
    """
    def __init__(self, realsense_server_path, output_folder, mediapipe_face_model_file, patient_image_path, log_table_path, visualize=False):
        self.mediapipe_face_model_file = mediapipe_face_model_file
        patient_img = face_recognition.load_image_file(patient_image_path)
        self.patients_encoding = face_recognition.face_encodings(patient_img)[0] # warning, it's a list!! That's why [0]

        self.realsense_server_path = realsense_server_path
        self.output_folder = output_folder
        self.log_table_path = log_table_path
        self.visualize = visualize

    def process_directories(self):
        """
        Processes directories containing .bag files, extracts facial movement data, and logs the process.
        Assumes .bag files are located in the deepest subdirectories of `realsense_server_path`.

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

                    self._append_row_and_save(log_table)

    
    def _realsense_day_extract(self, folder_name):
        """
        Extracts the date from a Realsense folder name, searching anywhere in the string.

        Parameters:
        folder_name (str): The name of the folder, which may contain a date in the format Year_month_day anywhere.

        Returns:
        str: The extracted date in the format "YYYY_MM_DD" , or None if no date is found.
        """
        # Updated regex to search for the date pattern anywhere in the string
        match = re.search(r"(\d{4}_\d{2}_\d{2})", folder_name)
        if match:
            return match.group(1)
        else:
            return None

    def _append_row_and_save(self, new_row_df):
        """
        Appends a new row to the log table and saves it. Creates a new log table if it doesn't exist.

        Parameters:
            new_row_df (pandas.DataFrame): DataFrame containing a single row to be appended to the log table.
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
        Processes a .bag file to extract forehead points, saves the outputs, and logs the process.

        Parameters:
        - path_bag (str): Path to the bag file to be processed.
        - output_file_path (str): Directory path where output files will be saved.

        Returns:
        - log_table (DataFrame): A pandas DataFrame containing the log of the operation, including paths, success/failure, and any errors encountered.

        The function attempts to extract facial vectors from the specified bag file using the provided MediaPipe face model and patient encodings. 
        It logs the process, including whether it succeeded or failed, and saves the extracted data to the specified output path. 
        In case of exceptions, it logs the error and marks the operation as failed.
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


class Synchronizer:
    """
    Synchronizes Realsense and WiSci data

    Attributes:
        mediapipe_face_model_file (str): Path to the MediaPipe face model file.
        patients_encoding (numpy.ndarray): Encoding of the patient's face from an image.
        wisci_server_path (str): Directory path where WiSci data is stored.
        output_folder (str): Directory path where synchronized data will be saved.
        log_table_path (str): Path to the CSV log table.
        log_table (pandas.DataFrame): DataFrame loaded from the CSV log table.
        visualize (bool): Flag to enable or disable visualization of the synchronization process.
    """
    def __init__(self, wisci_server_path, output_folder, mediapipe_face_model_file, patient_image_path, log_table_path, visualize=False):
        """
        Initializes the Synchronizer with paths and settings for data synchronization.

        Parameters:
            wisci_server_path (str): Path to the directory containing WiSci data.
            output_folder (str): Path to the directory where output data will be saved.
            mediapipe_face_model_file (str): Path to the MediaPipe face model file.
            patient_image_path (str): Path to the image file of the patient for face recognition.
            log_table_path (str): Path to the CSV file used as a log table.
            visualize (bool, optional): If True, enables visualization of the process. Defaults to False.
        """
        self.mediapipe_face_model_file = mediapipe_face_model_file
        patient_img = face_recognition.load_image_file(patient_image_path)
        self.patients_encoding = face_recognition.face_encodings(patient_img)[0] # warning, it's a list!! That's why [0]

        self.wisci_server_path = wisci_server_path
        self.output_folder = output_folder
        self.log_table_path = log_table_path
        # read the csv table
        self.log_table = pd.read_csv(log_table_path)
        self.visualize = visualize
    
    def synchronize_realsense_between_all_relevant_wisci_files(self):
        """
        Goes through the log table and synchronizes Realsense data with relevant WiSci files based on the date.

        Iterates over each row in the log table, skipping rows where the synchronization is done
        or where the facial point extraction has failed.
        For each relevant row, it finds matching WiSci files by date and updates the log table with these files.
        """

        # iterate over rows in the log table
        for index, row in self.log_table.iterrows():
            if row['failed'] == 1:
                continue
            # column 'path_wisci' has value (meaning it has already been synchronized)
            if pd.notnull(row['path_wisci']):
                continue

            possible_mat_files = self.find_relevant_mat_files(row['date']) # TODO: don't I need [0]?
            self.log_table.at[index, 'relevant_wisci_files'] = possible_mat_files
            self.log_table.to_csv(self.log_table_path, index=False)
    
    def find_relevant_mat_files(self, date):
        """
        Finds all .mat files relevant to a given date within the WiSci data directory.

        Parameters:
            date (str): The date for which to find relevant .mat files, in the format "YYYY_MM_DD".

        Returns:
            list: A list of paths to the relevant .mat files found.
            
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

    face_movement_extractor = FaceMovementExtractor(
        realsense_server_path=args.realsense_server_path, 
        output_folder=args.output_folder, 
        mediapipe_face_model_file='/home/mpicek/Downloads/face_landmarker.task', 
        patient_image_path="/home/mpicek/repos/master_project/data_synchronization/patient.png", 
        log_table_path="/home/mpicek/repos/master_project/processed2/table.csv", 
        visualize=True
    )

    face_movement_extractor.process_directories()

    synchronizer = Synchronizer(
        realsense_server_path=args.realsense_server_path, 
        output_folder=args.output_folder, 
        mediapipe_face_model_file='/home/mpicek/Downloads/face_landmarker.task', 
        patient_image_path="/home/mpicek/repos/master_project/data_synchronization/patient.png", 
        log_table_path="/home/mpicek/repos/master_project/processed2/table.csv", 
        visualize=True
    )

    synchronizer.synchronize_realsense_between_all_relevant_wisci_files()
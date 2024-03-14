import numpy as np
import pandas as pd
import os
from synchronization_utils import log_table_columns

class Synchronizer:
    """
    Synchronizes Realsense and WiSci data

    Attributes:
        wisci_server_path (str): Directory path where WiSci data is stored.
        output_folder (str): Directory path where synchronized data will be saved.
        log_table_path (str): Path to the CSV log table.
        log_table (pandas.DataFrame): DataFrame loaded from the CSV log table.
        visualize (bool): Flag to enable or disable visualization of the synchronization process.
    """
    def __init__(self, wisci_server_path, output_folder, log_table_path, visualize=False):
        """
        Initializes the Synchronizer with paths and settings for data synchronization.

        Parameters:
            wisci_server_path (str): Path to the directory containing WiSci data.
            output_folder (str): Path to the directory where output data will be saved.
            log_table_path (str): Path to the CSV file used as a log table.
            visualize (bool, optional): If True, enables visualization of the process. Defaults to False.
        """

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

            possible_mat_files = self.find_relevant_mat_files(row['date'])
            self.log_table.at[index, 'relevant_wisci_files'] = "TODO" # TODO: the cell has to be non-empty in order to write there a list of values
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
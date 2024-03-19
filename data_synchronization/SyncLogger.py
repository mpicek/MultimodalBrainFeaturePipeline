import os
import pandas as pd

class SyncLogger:
    def __init__(self, log_table_path):
        """
        Initialize the logger, loading an existing log file if present.
        
        Parameters:
        - filepath: The path where the log file will be saved. If the file exists, it's loaded.
        """
        self.log_table_path = log_table_path
        self.columns = [
            # the following are for the extraction
            'path_bag', # path to the bag file processed
            'output_path', # path to the output folder
            'sync_strategy', # acc, LED
            'date', # date of the recording (from the bag filename)
            'acc_extraction_failed', # 0/1
            'acc_avg_quality', # average quality of the frames
            'first_frame_face_detected',
            'face_location',
            'face_patient_distance',
            'Exception5000',
            'unknown_error_extraction',
            'relevant_wisci_files',
            # and these are for the synchronization
            'path_wisci',
            'normalized_corr',
            'lag',
            'fps_bag',
            'unknown_error_synchronization',
            'peaks_per_million',
            'best_second_largest_corr_peak',
            'second_best_wisci_corr',
            'second_best_wisci_peaks_per_million',
            'second_best_wisci_path',
            'synchronization_failed',
        ]

        if os.path.exists(log_table_path):
            self.log_df = pd.read_csv(log_table_path)
        else:
            self.log_df = pd.DataFrame(columns=self.columns)


    def process_new_file(self, path_bag):
        """
        Start logging a new file by appending a row with the filename.
        
        Parameters:
        - filename: The name of the file being processed.
        """
        if not self.log_df['path_bag'].str.contains(path_bag).any():

            filled_columns = {column: None for column in self.columns}
            filled_columns['path_bag'] = path_bag

            new_row = pd.DataFrame([filled_columns])
            self.log_df = pd.concat([self.log_df, new_row], ignore_index=True)
        
        else:
            raise ValueError(f"File {path_bag} already exists in the log.") # TODO: maybe specific exception

    
    def update_log(self, path_bag, column, value):
        """
        Update the log for a specific file and column.
        
        Parameters:
        - filename: The name of the file to update.
        - column: The column to update.
        - value: The new value for the column.
        """
        # Find the row index for the given filename. Assumes filename is unique for each row.
        row_index = self.log_df[self.log_df['path_bag'] == path_bag].index[0]
        self.log_df.at[row_index, column] = value
    
    def get_value(self, path_bag, column):
        """
        Get the value of a specific column for a specific file.
        
        Parameters:
        - filename: The name of the file to get the value for.
        - column: The column to get the value for.
        
        Returns:
        - The value of the column for the given file.
        """
        # Find the row index for the given filename. Assumes filename is unique for each row.
        row_index = self.log_df[self.log_df['path_bag'] == path_bag].index[0]
        return self.log_df.at[row_index, column]
    

    
    def save_to_csv(self):
        """
        Save the log to a CSV file.
        """
        self.log_df.to_csv(self.log_table_path, index=False)

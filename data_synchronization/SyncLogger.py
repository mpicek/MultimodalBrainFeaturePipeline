import os
import pandas as pd

class FileAlreadySynchronized(Exception):
    pass

class SyncLogger:
    def __init__(self, log_table_path):
        """
        Initialize the logger, loading an existing log file if present.
        
        Parameters:
        - filepath: The path where the log file will be saved. If the file exists, it's loaded.
        """
        self.log_table_path = log_table_path
        self.columns = [
            # and these are for the synchronization
            'video_name',
            'video_length',
            'path_ecog',
            'corr',
            'lag',
            'sync_error_msg',
            'best_second_largest_corr_peak',
            'sync_failed',
            'avg_quality',
        ]

        if os.path.exists(log_table_path):
            self.original_df = pd.read_csv(log_table_path)
        else:
            self.original_df = pd.DataFrame(columns=self.columns)
        
        self.log_df = None


    def process_new_file(self, video_name):
        """
        Start logging a new file by appending a row with the filename.
        
        Parameters:
        - filename: The name of the file being processed.
        """
        if not self.original_df['video_name'].str.contains(video_name).any():

            filled_columns = {column: None for column in self.columns}
            filled_columns['video_name'] = video_name

            self.log_df = pd.DataFrame([filled_columns])
            # self.log_df = pd.concat([self.log_df, new_row], ignore_index=True)
        
        else:
            raise FileAlreadySynchronized(f"File {video_name} already exists in the log.")

    
    def update_log(self, video_name, column, value):
        """
        Update the log for a specific file and column.
        
        Parameters:
        - filename: The name of the file to update.
        - column: The column to update.
        - value: The new value for the column.
        """
        # Find the row index for the given filename. Assumes filename is unique for each row.
        row_index = self.log_df[self.log_df['video_name'] == video_name].index[0]
        self.log_df.at[row_index, column] = value
    
    def get_value(self, video_name, column):
        """
        Get the value of a specific column for a specific file.
        
        Parameters:
        - filename: The name of the file to get the value for.
        - column: The column to get the value for.
        
        Returns:
        - The value of the column for the given file.
        """
        # Find the row index for the given filename. Assumes filename is unique for each row.
        row_index = self.log_df[self.log_df['video_name'] == video_name].index[0]
        return self.log_df.at[row_index, column]
    

    
    def save_to_csv(self):
        """
        Save the log to a CSV file.
        """
        if os.path.exists(self.log_table_path):
            self.original_df = pd.read_csv(self.log_table_path)
        else:
            self.original_df = pd.DataFrame(columns=self.columns)

        self.original_df = pd.concat([self.original_df, self.log_df], ignore_index=True)

        self.original_df.to_csv(self.log_table_path, index=False)

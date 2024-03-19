import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from synchronization_utils import preprocess_accelerometer_data, preprocess_video_signals
from accelerometer import get_accelerometer_data, GettingAccelerometerDataFailed
import traceback

class Synchronizer:
    """
    Synchronizes Realsense and WiSci data

    Attributes:
        wisci_server_path (str): Directory path where WiSci data is stored.
        output_folder (str): Directory path where synchronized data will be saved.
        log_table_path (str): Path to the CSV log table.
        log_table (pandas.DataFrame): DataFrame loaded from the CSV log table.
        box_size (int): Size of the box for smoothing the accelerometer data.
        visualize (bool): Flag to enable or disable visualization of the synchronization process.
    """
    def __init__(self, wisci_server_path, output_folder, log_table_path, box_size, visualize=False):
        """
        Initializes the Synchronizer with paths and settings for data synchronization.

        Parameters:
            wisci_server_path (str): Path to the directory containing WiSci data.
            output_folder (str): Path to the directory where output data will be saved.
            log_table_path (str): Path to the CSV file used as a log table.
            box_size (int): Size of the box for smoothing the accelerometer data.
            visualize (bool, optional): If True, enables visualization of the process. Defaults to False.
        """

        self.wisci_server_path = wisci_server_path
        self.output_folder = output_folder
        self.log_table_path = log_table_path
        # read the csv table
        self.log_table = pd.read_csv(log_table_path)
        self.box_size = box_size
        self.visualize = visualize

    def sync_and_optimize_freq(self, accelerometer_data, forehead_points, quality, cam2face, duration):
        # trying different frequencies as wisci and realsense are both imprecise in their sampling
        # frequencies so I adjust it like this - we find the freq that gets the best result :)
        freqs = np.linspace(29.3, 30.5, 50)
        best_corr = -1
        best_lag = 0
        best_freq = freqs[0]
        best_total_peaks = 0
        best_second_largest_corr_peak = 0
        for freq in freqs:
            video_gradient_smoothed, quality_resampled = preprocess_video_signals(forehead_points, quality, cam2face, duration, freq, box_size=self.box_size)
            corr, lag, total_peaks, second_largest_corr_peak = self.sync_wisci_and_video(accelerometer_data, video_gradient_smoothed, quality_resampled)
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
                best_freq = freq
                best_total_peaks = total_peaks
                best_second_largest_corr_peak = second_largest_corr_peak
        
        return best_corr, best_lag, best_freq, best_total_peaks, best_second_largest_corr_peak
    
    def sync_with_best_wisci_file(self, possible_wisci_files, log_table_index, log_table_row):
        best_corr_per_wisci_file = []
        best_lag_per_wisci_file = []
        best_freq_per_wisci_file = []
        best_total_peaks_per_wisci_file = []
        best_second_largest_corr_peak_per_wisci_file = []

        # go over wisci files
        for wisci_file in possible_wisci_files:
            data_acc = get_accelerometer_data(wisci_file) # can throw exception that will propagate up
            accelerometer_data = preprocess_accelerometer_data(data_acc, self.box_size)

            # open the relevant files
            forehead_points = np.load(os.path.join(log_table_row['output_path'], 'forehead_points.npy'))
            quality = np.load(os.path.join(log_table_row['output_path'], 'quality.npy'))
            cam2face = np.load(os.path.join(log_table_row['output_path'], 'cam2face.npy'))
            duration = np.load(os.path.join(log_table_row['output_path'], 'duration.npy'))

            if len(forehead_points) > len(accelerometer_data):
                # I suppose that the whole realsense recording has to be in the wisci recording
                # (and not just part of it)
                best_corr_per_wisci_file.append(0)
                best_lag_per_wisci_file.append(0)
                best_freq_per_wisci_file.append(0)
                best_total_peaks_per_wisci_file.append(0)
                continue

            best_corr, best_lag, best_freq, best_total_peaks, best_second_largest_corr_peak = self.sync_and_optimize_freq(accelerometer_data, forehead_points, quality, cam2face, duration)
            
            best_corr_per_wisci_file.append(best_corr)
            best_lag_per_wisci_file.append(best_lag)
            best_freq_per_wisci_file.append(best_freq)
            best_total_peaks_per_wisci_file.append(best_total_peaks)
            best_second_largest_corr_peak_per_wisci_file.append(best_second_largest_corr_peak)

        best_corr_per_wisci_file = np.array(best_corr_per_wisci_file)
        index_of_best_wisci_file = np.argmax(best_corr_per_wisci_file)
        max_corr = best_corr_per_wisci_file[index_of_best_wisci_file]
        lag = best_lag_per_wisci_file[index_of_best_wisci_file]
        freq = best_freq_per_wisci_file[index_of_best_wisci_file]
        index_of_second_best_wisci_file = np.argsort(best_corr_per_wisci_file)[-2]

        self.log_table.at[log_table_index, 'path_wisci'] = possible_wisci_files[index_of_best_wisci_file]
        self.log_table.at[log_table_index, 'normalized_corr'] = max_corr
        self.log_table.at[log_table_index, 'lag'] = lag
        self.log_table.at[log_table_index, 'fps_bag'] = freq
        self.log_table.at[log_table_index, 'peaks_per_million'] = best_total_peaks_per_wisci_file[index_of_best_wisci_file]/len(accelerometer_data) * 1000000
        self.log_table.at[log_table_index, 'best_second_largest_corr_peak'] = best_second_largest_corr_peak_per_wisci_file[index_of_best_wisci_file]
        self.log_table.at[log_table_index, 'second_best_wisci_corr'] = best_corr_per_wisci_file[index_of_second_best_wisci_file]
        self.log_table.at[log_table_index, 'second_best_wisci_peaks_per_million'] = best_total_peaks_per_wisci_file[index_of_second_best_wisci_file]/len(accelerometer_data) * 1000000
        self.log_table.at[log_table_index, 'second_best_wisci_path'] = possible_wisci_files[index_of_second_best_wisci_file]
        self.log_table.at[log_table_index, 'synchronization_failed'] = 0
        self.log_table.to_csv(self.log_table_path, index=False)

    def process_all_realsense_recordings(self):
        """
        Goes through the log table and synchronizes Realsense data with relevant WiSci files based on the date.

        Iterates over each row in the log table, skipping rows where the synchronization is done
        or where the facial point extraction has failed.
        For each relevant row, it finds matching WiSci files by date and updates the log table with these files.
        """

        # iterate over rows in the log table (aka over all bag files we want to synchronize)
        for index, row in self.log_table.iterrows():
            try:
                print(f"{index + 1}/{len(list(self.log_table.iterrows()))}: Bag being processed: {row['path_bag']}")
                if row['failed'] == 1:
                    print("There was an error in extraction, skipping synchronization.")
                    continue
                # if the columns 'path_wisci' has no value means it has already been synchronized
                if pd.notnull(row['path_wisci']):
                    print("Already synchronized")
                    print(row['path_wisci'])
                    continue

                possible_wisci_files = self.find_relevant_wisci_files(row['date'])
                self.log_table.at[index, 'relevant_wisci_files'] = "TODO" # TODO: the cell has to be non-empty in order to write there a list of values
                self.log_table.at[index, 'relevant_wisci_files'] = possible_wisci_files
                self.log_table.to_csv(self.log_table_path, index=False)
            
                self.sync_with_best_wisci_file(possible_wisci_files, index, row)

                if index == 0:
                    self.visualize_synchronized(row['path_bag'])
            
            except GettingAccelerometerDataFailed as e:
                self.log_table.at[index, 'unknown_error_synchronization'] = "GettingAccelerometerDataFailed"
                self.log_table.at[index, 'synchronization_failed'] = 1
                self.log_table.to_csv(self.log_table_path, index=False)

            except Exception as e:
                print(e)
                self.log_table.at[index, 'unknown_error_synchronization'] = traceback.format_exc()
                self.log_table.at[index, 'synchronization_failed'] = 1
                self.log_table.to_csv(self.log_table_path, index=False)


    def find_relevant_wisci_files(self, date):
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

        possible_wisci_files = []
        for folder in os.listdir(self.wisci_server_path):
            if date not in folder: # skip if the date is not in the folder's name
                continue

            # we found a folder from the correct date
            day_folder = os.path.join(self.wisci_server_path, folder)
            # now find wisci folder
            for day_subfolder in os.listdir(day_folder):
                # if includes wisci or wimagine (case insensitive)
                if 'wisci' not in day_subfolder.lower() and 'wimagine' not in day_subfolder.lower():
                    continue

                wisci_folder = os.path.join(day_folder, day_subfolder)

                # now in wisci folder, there is many subfolders, each containing just one .mat file,
                # let's find those .mat files
                for root, _, files in os.walk(wisci_folder): # returns all possible paths in the tree
                    for file in files:
                        if file.endswith('.mat'):
                            wisci_file = os.path.join(root, file)
                            possible_wisci_files.append(wisci_file)
        
        return possible_wisci_files
    
    def visualize_synchronized(self, path_bag):
        # get the row from the log table
        row = self.log_table[self.log_table['path_bag'] == path_bag]
        if len(row) == 0:
            print("No such path in the log table.")
            return
        row = row.iloc[0]
        # load the data
        data_acc = get_accelerometer_data(row['path_wisci'])
        accelerometer_data = preprocess_accelerometer_data(data_acc, self.box_size)
        forehead_points = np.load(os.path.join(row['output_path'], 'forehead_points.npy'))
        quality = np.load(os.path.join(row['output_path'], 'quality.npy'))
        cam2face = np.load(os.path.join(row['output_path'], 'cam2face.npy'))
        duration = np.load(os.path.join(row['output_path'], 'duration.npy'))
        video_gradient_smoothed, quality_resampled = preprocess_video_signals(forehead_points, quality, cam2face, duration, row['fps_bag'], box_size=self.box_size)
        corr, lag, _, second_largest_corr_peak = self.sync_wisci_and_video(accelerometer_data, video_gradient_smoothed, quality_resampled, overwrite_visualization=True)
    
    def sync_wisci_and_video(self, accelerometer_data, video, quality, overwrite_visualization=False):

        corr_accumulated = None

        arr = []
        for i in range(3):
            sig1 = accelerometer_data[:, i]
            for j in range(3):
                sig2 = video[:, j]

                shifted_sig1 = sig1 - np.mean(sig1)
                shifted_sig2 = sig2 - np.mean(sig2)
                normalized_sig1 = shifted_sig1 / (np.std(shifted_sig1))
                normalized_sig2 = shifted_sig2 / (np.std(shifted_sig2))

                sig1 = normalized_sig1
                sig2 = normalized_sig2
                cut_by_smoothing_beginning = (self.box_size - 1)//2
                cut_by_smoothing_end = -(self.box_size - 1)//2
                sig2 = sig2 * quality[cut_by_smoothing_beginning:cut_by_smoothing_end][cut_by_smoothing_beginning:cut_by_smoothing_end][cut_by_smoothing_beginning:cut_by_smoothing_end]

                # we use "valid" because we suppose that the realsense video is whole in the wisci recording (not just a part of it)
                correlation = signal.correlate(sig1, sig2, mode="valid")
                normalization_factor = len(sig2) # not this: min(len(sig1), len(sig2)) because the video has to fit there in the recording
                correlation /= normalization_factor
                lags = signal.correlation_lags(sig1.size, sig2.size, mode="valid")
                if corr_accumulated is None:
                    corr_accumulated = np.abs(correlation)
                else:
                    corr_accumulated += np.abs(correlation)
                lag = lags[np.argmax(np.abs(correlation))]
                arr.append([i, j, np.max(np.abs(correlation)), lag, lag/(60*30)])


                if self.visualize or overwrite_visualization:
                    print(lag)
                    print(i, j)
                    print(np.max(np.abs(correlation)))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=np.arange(len(np.abs(correlation))), y=np.abs(correlation), mode='lines', name='correlation', line=dict(color='red')))
                    fig.show()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=np.arange(len(sig1)), y=sig1, mode='lines', name='correlation', line=dict(color='red')))
                
                    # we computed abs(correlation), to visualize the signals, let's align them
                    # (if they were anticorrelated, just put minus before the signal)
                    if correlation[np.argmax(np.abs(correlation))] > 0:
                        fig.add_trace(go.Scatter(x=np.arange(len(sig2)) + lag, y=sig2, mode='lines', name='correlation', line=dict(color='blue')))
                    else:
                        fig.add_trace(go.Scatter(x=np.arange(len(sig2)) + lag, y=-sig2, mode='lines', name='correlation', line=dict(color='blue')))
                    fig.show()


        peaks, _ = find_peaks(np.abs(corr_accumulated), height=0.5*np.max(np.abs(corr_accumulated)), distance=30*2) # TODO: fix distance by the freq
        second_largest_corr_peak = 0
        if len(peaks) < 2:
            second_largest_corr_peak = 0
        else:
            peak_values = np.abs(corr_accumulated)[peaks]
            sorted_unique_peak_values = np.sort(np.unique(peak_values))
            second_largest_corr_peak = sorted_unique_peak_values[-2]

        if self.visualize or overwrite_visualization:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(len(corr_accumulated)), y=corr_accumulated, mode='lines', name='correlation', line=dict(color='blue')))
            fig.show()

            arr = np.array(arr)
            # make it a pandas df
            df = pd.DataFrame(arr, columns=["i", "j", "corr", "lag", "time"])
            # sort by correlation
            df = df.sort_values("corr", ascending=False)
            print(lags[np.argmax(corr_accumulated)])
            print(df)

        return np.max(corr_accumulated), lags[np.argmax(corr_accumulated)], len(peaks), second_largest_corr_peak
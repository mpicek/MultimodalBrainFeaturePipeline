import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from synchronization_utils import preprocess_accelerometer_data, preprocess_video_signals, extract_movement_from_dlc_csv
from SyncLogger import SyncLogger, FileAlreadySynchronized
from wisci_utils import get_accelerometer_data, GettingAccelerometerDataFailed, find_corresponding_wisci
import traceback
from scipy.signal import resample
from tqdm import tqdm

class Synchronizer:
    """
    Synchronizes Realsense and WiSci data

    Attributes:
        wisci_server_path (str): Directory path where WiSci data is stored.
        output_folder (str): Directory path where synchronized data will be saved.
        log_table_path (str): Path to the CSV log table.
        log_table (pandas.DataFrame): DataFrame loaded from the CSV log table.
        sigma (int): Sigma of the Gaussian filter for smoothing.
        visualize (bool): Flag to enable or disable visualization of the synchronization process.
    """
    def __init__(self, dlc_csv_folder, wisci_server_path, log_table_path, sigma_video, sigma_wisci, sync_images_folder=None, visualize=False):
        """
        Initializes the Synchronizer with paths and settings for data synchronization.

        Parameters:
            wisci_server_path (str): Path to the directory containing WiSci data.
            output_folder (str): Path to the directory where output data will be saved.
            log_table_path (str): Path to the CSV file used as a log table.
            sigma (int): Sigma of the Gaussian filter for smoothing.
            visualize (bool, optional): If True, enables visualization of the process. Defaults to False.
        """

        self.wisci_server_path = wisci_server_path
        self.log_table_path = log_table_path
        self.log = SyncLogger(log_table_path)
        self.sigma_video = sigma_video
        self.sigma_wisci = sigma_wisci
        self.visualize = visualize
        self.dlc_csv_folder = dlc_csv_folder
        self.dlc_suffix = 'DLC_resnet50_UP2_movement_syncApr15shuffle1_525000.csv'
        self.sync_images_folder = sync_images_folder

        if self.sync_images_folder is not None and not os.path.exists(self.sync_images_folder):
            os.makedirs(self.sync_images_folder)

    def sync_and_optimize_freq(self, accelerometer_data, forehead, accelerometer_duration, video_duration, quality, original_mp4_basename=None):
        # trying different frequencies as wisci and realsense are both imprecise in their sampling
        # frequencies so I adjust it like this - we find the freq that gets the best result :)
        # print("SYNCING. ACCELEROMETR DATA LENGTH:", len(accelerometer_data))
        # print("FOREHEAD POINTS LENGTH:", len(forehead_points))

        wisci_freq = len(accelerometer_data) / accelerometer_duration
        video_freq = len(forehead) / video_duration
        print("Frequencies - Wisci: ", wisci_freq, ", Video: ", video_freq)
        print(f"Video frames (with original FPS): {len(forehead)}")

        # upsample the video to the wisci frequency
        new_num_samples = int(len(forehead) * (wisci_freq / video_freq))

        best_corr = -1
        best_lag = 0
        best_total_peaks = 0
        best_second_largest_corr_peak = 0
        best_corr_accumulated = None
        best_n_samples = 0
        corrs = []

        best_signal = None

        # TODO: adjust the range and number of tries based on duration
        for n_samples in tqdm(np.linspace(new_num_samples - 1500, new_num_samples + 1500, 150)):

            resampled_video = resample(forehead, int(n_samples))
            quality_resampled = resample(quality, int(n_samples))

            corr, lag, total_peaks, second_largest_corr_peak, corr_accumulated = self.sync_wisci_and_video(accelerometer_data, resampled_video, quality_resampled, original_mp4_basename)
            corrs.append(corr)
            if corr > best_corr:
                best_corr, best_lag, best_total_peaks, best_second_largest_corr_peak, best_corr_accumulated = corr, lag, total_peaks, second_largest_corr_peak, corr_accumulated
                best_signal = resampled_video
                best_n_samples = n_samples
    
        resampled_video = resample(forehead, int(best_n_samples))
        quality_resampled = resample(quality, int(best_n_samples))
        # _ = self.sync_wisci_and_video(accelerometer_data, resampled_video, quality_resampled, original_mp4_basename, overwrite_visualization=True)
        if self.sync_images_folder is not None and original_mp4_basename is not None:
            plt.figure()
            plt.plot(best_corr_accumulated)
            plt.savefig(os.path.join(self.sync_images_folder, original_mp4_basename[:-4] + "_corr.png"))

            around = 15*585 # show 15s before (and 15s after) the log
            around = min(around, best_lag) # shorten it from the left side
            around = min(around, len(accelerometer_data) - best_lag, len(best_signal)) # shorten it from the right side
            fig, ax = plt.subplots(figsize=(15, 5), dpi=200)
            ax.plot(np.arange(2*around), accelerometer_data[best_lag-around:best_lag+around, 0], color='red', label='Wisci')
            ax.plot(np.arange(around) + around, best_signal[:around, 0], color='blue', label='From video')
            ax.plot(np.arange(585), np.zeros((585,)), color='black', label='1s stretch', linewidth=2)
            plot_width = 2*around  # Adjust as needed

            # Set grid spacing based on the plot width
            grid_width = 585 / (1000 / 150)  # Adjust as needed

            # Add grid lines with predefined width
            ax.grid(True, linewidth=0.5, linestyle='--', color='gray', which='both', alpha=0.5)

            # Set x ticks with predefined width
            ax.set_xticks(np.arange(0, plot_width, grid_width))
            
            ax.legend()
            new_labels = [str(int(label) + best_lag - around) for label in ax.get_xticks()]
            ax.set_xticklabels(new_labels, fontsize=4, rotation=90)
            plt.savefig(os.path.join(self.sync_images_folder, original_mp4_basename[:-4] + "_beginning.png"))



            length = 120*585# first two minutes of the signal if possible
            # now we shorten the signal if it is too long (otherwise we would get an error due to inconsistent dimensions)
            length = min(120*585, len(best_signal[:length, 1]), len(accelerometer_data[best_lag:best_lag+length, 1])) 
            _, ax = plt.subplots(figsize=(15, 5), dpi=200)
            ax.plot(np.arange(length), accelerometer_data[best_lag:best_lag+length, 1], color='red', label='Wisci')
            ax.plot(np.arange(length), best_signal[:length, 1], color='blue', label='From video')
            ax.plot(np.arange(585), np.zeros((585,)), color='black', label='1s stretch', linewidth=2)
            ax.legend()
            plt.savefig(os.path.join(self.sync_images_folder, original_mp4_basename[:-4] + "_two_mins.png"))

            _, ax = plt.subplots(figsize=(15, 5), dpi=200)
            ax.plot(np.arange(len(accelerometer_data)), accelerometer_data[:, 1], color='red', label='Wisci')
            ax.plot(np.arange(len(best_signal)) + best_lag, best_signal[:, 1], color='blue', label='From video')
            ax.plot(np.arange(585), np.zeros((585,)), color='black', label='1s stretch', linewidth=2)
            ax.legend()
            plt.savefig(os.path.join(self.sync_images_folder, original_mp4_basename[:-4] + "_whole.png"))

        return best_corr, best_lag, best_n_samples, best_total_peaks, best_second_largest_corr_peak
    

    def sync_with_movement(self, csv_full_path, wisci_file, log=True, output_path_manual=None):

        csv_basename = os.path.basename(csv_full_path)
        original_mp4_basename = csv_basename[:-len(self.dlc_suffix)] + ".mp4"

        accelerometer_data, accelerometer_duration = get_accelerometer_data(wisci_file) # can throw exception that will propagate up

        accelerometer_data = preprocess_accelerometer_data(accelerometer_data, self.sigma_wisci)

        forehead, quality = extract_movement_from_dlc_csv(csv_full_path)
        video_duration = np.load(csv_full_path[:-len(self.dlc_suffix)] + '_duration.npy')

        if video_duration > 15: # process videos that are at least 15s long

            forehead = preprocess_video_signals(forehead, sigma=self.sigma_video)

            best_corr, best_lag, best_n_samples, best_total_peaks, best_second_largest_corr_peak = self.sync_and_optimize_freq(accelerometer_data, forehead, accelerometer_duration, video_duration, quality, original_mp4_basename)
            print(f"Just analyzed file: {original_mp4_basename}")
            print(f"\tBest correlation: {best_corr}\n\tLag: {best_lag}\n\tPeaks: {best_total_peaks-1}\n\tSecond largest peak: {best_second_largest_corr_peak}")
            sync_failed = 0

        else:
            best_corr, best_lag, best_n_samples, best_total_peaks, best_second_largest_corr_peak = -1, -1, -1, -1, -1

            sync_failed = 1
            self.log.update_log(original_mp4_basename, 'mp4_length', video_duration)
            self.log.update_log(original_mp4_basename, 'sync_error_msg', "Video too short (under 15s).")


        if log:
            self.log.update_log(original_mp4_basename, 'mp4_length', video_duration)
            self.log.update_log(original_mp4_basename, 'frames', best_n_samples)
            self.log.update_log(original_mp4_basename, 'path_wisci', wisci_file)
            self.log.update_log(original_mp4_basename, 'corr', best_corr)
            self.log.update_log(original_mp4_basename, 'lag', best_lag)
            self.log.update_log(original_mp4_basename, 'additional_peaks_per_million', (best_total_peaks-1)/len(accelerometer_data) * 1000000)
            self.log.update_log(original_mp4_basename, 'best_second_largest_corr_peak', best_second_largest_corr_peak)
            self.log.update_log(original_mp4_basename, 'sync_failed', sync_failed)
        else:
            # write the logs into a file logs_manual.txt in the output_path_manual (but if the file exists, append to it)
            with open(os.path.join(output_path_manual, 'logs_manual.txt'), 'a') as f:
                f.write(f"Path to the best WiSci file: {wisci_file}\n")
                f.write(f"Normalized correlation: {best_corr}\n")
                f.write(f"Number of frames: {best_n_samples}\n")
                f.write(f"Lag: {best_lag}\n")
                f.write(f"Peaks per million: {(best_total_peaks-1)/len(accelerometer_data) * 1000000}\n")
                f.write(f"Second largest correlation peak: {best_second_largest_corr_peak}\n")
                f.write(f"Synchronization failed: 0\n")


    def sync_mp4_folder(self):
        """
        Goes through the log table and synchronizes Realsense data with relevant WiSci files based on the date.

        Iterates over each row in the log table, skipping rows where the synchronization is done
        or where the facial point extraction has failed.
        For each relevant row, it finds matching WiSci files by date and updates the log table with these files.
        """

        processed_files = os.listdir(self.dlc_csv_folder)
        movement_csv_files_basenames = [os.path.basename(file) for file in processed_files if file.endswith('.csv')]

        for path_csv_basename in movement_csv_files_basenames:
            original_mp4_basename = path_csv_basename[:-len(self.dlc_suffix)] + ".mp4"

            try:
                self.log.process_new_file(original_mp4_basename)
            except FileAlreadySynchronized as e:
                print(e)
                print(f"Skipping {original_mp4_basename}")
                continue

            print("Processing: ", original_mp4_basename)
            wisci_file = find_corresponding_wisci(original_mp4_basename, self.wisci_server_path)
            try:
                self.sync_with_movement(os.path.join(self.dlc_csv_folder, path_csv_basename), wisci_file)
            except GettingAccelerometerDataFailed as e:
                print(f"Getting accelerometer data failed for {wisci_file}. Skipping the file.")
                self.log.update_log(original_mp4_basename, 'sync_error_msg', "GettingAccelerometerDataFailed")
                self.log.update_log(original_mp4_basename, 'sync_failed', 1)
            except Exception as e:
                print(traceback.format_exc())
                self.log.update_log(original_mp4_basename, 'sync_error_msg', traceback.format_exc())
                self.log.update_log(original_mp4_basename, 'sync_failed', 1)
            finally:
                self.log.save_to_csv()

    
    def visualize_synchronized(self, path_bag):
        # get the row from the log table

        output_path = self.log.get_value(path_bag, 'output_path')
        # load the data
        data_acc = get_accelerometer_data(self.log.get_value(path_bag, 'path_wisci'))
        accelerometer_data = preprocess_accelerometer_data(data_acc, self.sigma)
        forehead_points = np.load(os.path.join(output_path, 'forehead_points.npy'))
        quality = np.load(os.path.join(output_path, 'quality.npy'))
        quality = np.load(os.path.join(output_path, 'quality.npy'))
        cam2face = np.load(os.path.join(output_path, 'cam2face.npy'))
        duration = np.load(os.path.join(output_path, 'duration.npy'))
        video_gradient_smoothed, quality_resampled = preprocess_video_signals(forehead_points, quality, cam2face, duration, self.log.get_value(path_bag, 'fps_bag'), sigma=self.sigma)
        corr, lag, _, second_largest_corr_peak = self.sync_wisci_and_video(accelerometer_data, video_gradient_smoothed, quality_resampled, overwrite_visualization=True)
    
    def sync_wisci_and_video(self, accelerometer_data, video, quality, original_mp4_basename=None, overwrite_visualization=False):

        corr_accumulated = None

        arr = []
        for i in range(3):
            sig1 = accelerometer_data[:, i]
            for j in range(2):
                sig2 = video[:, j]

                sig2 = sig2 * quality

                # we use "valid" because we suppose that the realsense video is whole in the wisci recording (not just a part of it)
                correlation = signal.correlate(sig1, sig2, mode="valid")
                lags = signal.correlation_lags(sig1.size, sig2.size, mode="valid")

                # IMPORTANT to normalize this.
                # If we have some strong movement but it does not correlate with the video (for example head moving back and forth),
                # then the correlation will be high everywhere. But if there is some small movement but correlates highly (overall
                # corr is small but the peak is very prominent), then this might be overpowered by the first case.
                correlation_normalized = correlation

                if corr_accumulated is None:
                    corr_accumulated = np.abs(correlation_normalized)
                else:
                    corr_accumulated += np.abs(correlation_normalized)
                lag = lags[np.argmax(np.abs(correlation_normalized))]
    
                arr.append([i, j, lag])

                if (self.visualize or overwrite_visualization) and len(sig1) < 1000000:
                    print(i, j)
                    print("Lag: ", lag)
                    print("correlation: ", np.max(np.abs(correlation)))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=np.arange(len(np.abs(correlation_normalized))), y=np.abs(correlation_normalized), mode='lines', name='correlation', line=dict(color='red')))
                    fig.show()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=np.arange(len(sig1)), y=sig1, mode='lines', name='correlation', line=dict(color='red')))
                
                    # we computed abs(correlation), to visualize the signals, let's align them
                    # (if they were anticorrelated, just put minus before the signal)
                    if correlation_normalized[np.argmax(np.abs(correlation_normalized))] > 0:
                        fig.add_trace(go.Scatter(x=np.arange(len(sig2)) + lag, y=sig2, mode='lines', name='correlation', line=dict(color='blue')))
                    else:
                        fig.add_trace(go.Scatter(x=np.arange(len(sig2)) + lag, y=-sig2, mode='lines', name='correlation', line=dict(color='blue')))
                    fig.show()


        peaks, _ = find_peaks(corr_accumulated, height=0.5*np.max(corr_accumulated), distance=585*2) # TODO: fix distance by the freq
        second_largest_corr_peak = 0
        if len(peaks) < 2:
            second_largest_corr_peak = 0
        else:
            peak_values = np.abs(corr_accumulated)[peaks]
            sorted_unique_peak_values = np.sort(np.unique(peak_values))
            second_largest_corr_peak = sorted_unique_peak_values[-2]

        if (self.visualize or overwrite_visualization) and len(sig1) < 1000000:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(len(corr_accumulated)), y=corr_accumulated, mode='lines', name='correlation', line=dict(color='blue')))
            fig.show()
    
            arr = np.array(arr)
            # make it a pandas df
            df = pd.DataFrame(arr, columns=["i", "j", "lag"])
            # sort by correlation

        return np.max(corr_accumulated), lags[np.argmax(corr_accumulated)], len(peaks), second_largest_corr_peak, corr_accumulated
    

if __name__=="__main__":

    # data_folder = "/media/vita-w11/T71/UP2001/"
    data_folder = '/home/vita-w11/mpicek/data/'
    dlc_csv_folder = os.path.join(data_folder, 'mp4')
    wisci_server_path = os.path.join(data_folder, 'WISCI')
    log_table_path = os.path.join(data_folder, 'sync_log.csv')
    sync_images_folder = os.path.join(data_folder, 'sync_images')




    data_folder = '/media/vita-w11/T72/UP2001/bags/subset/'
    dlc_csv_folder = os.path.join(data_folder, 'mp4')
    wisci_server_path = '/media/vita-w11/T72/UP2001/WISCI/'
    log_table_path = os.path.join(data_folder, 'sync_log.csv')
    sync_images_folder = os.path.join(data_folder, 'sync_images')
    sigma_video = 10
    sigma_wisci = 200
    visualize = False

    synchronizer = Synchronizer(dlc_csv_folder, wisci_server_path, log_table_path, sigma_video, sigma_wisci, sync_images_folder, visualize)
    synchronizer.sync_mp4_folder()
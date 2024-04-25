import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from synchronization_utils import preprocess_accelerometer_data, preprocess_video_signals, extract_movement_from_dlc_csv
from SyncLogger import SyncLogger, FileAlreadySynchronized
from wisci_utils import get_accelerometer_data, GettingAccelerometerDataFailed, get_LED_data, find_corresponding_wisci
import traceback
from scipy.signal import resample
from tqdm import tqdm
import argparse

class LedSynchronizer:
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
    def __init__(self, mp4_folder, led_signals_folder, wisci_server_path, log_table_path, sync_images_folder=None, visualize=False):
        """
        Initializes the Synchronizer with paths and settings for data synchronization.

        Parameters:
            wisci_server_path (str): Path to the directory containing WiSci data.
            output_folder (str): Path to the directory where output data will be saved.
            log_table_path (str): Path to the CSV file used as a log table.
            sigma (int): Sigma of the Gaussian filter for smoothing.
            visualize (bool, optional): If True, enables visualization of the process. Defaults to False.
        """
        self.mp4_folder = mp4_folder # path to the mp4 files WITH THE SAVED DURATION IN .NPY FILE!
        self.led_signals_folder = led_signals_folder # path to the folder with the LED signals
        self.wisci_server_path = wisci_server_path
        self.log_table_path = log_table_path
        self.log = SyncLogger(log_table_path)
        self.visualize = visualize
        self.sync_images_folder = sync_images_folder

        if self.sync_images_folder is not None and not os.path.exists(self.sync_images_folder):
            os.makedirs(self.sync_images_folder)

    def sync_and_optimize_freq(self, accelerometer_data, forehead, accelerometer_duration, video_duration, original_mp4_basename=None):
        # trying different frequencies as wisci and realsense are both imprecise in their sampling
        # frequencies so I adjust it like this - we find the freq that gets the best result :)
        # print("SYNCING. ACCELEROMETR DATA LENGTH:", len(accelerometer_data))
        # print("FOREHEAD POINTS LENGTH:", len(forehead_points))

        wisci_freq = len(accelerometer_data) / accelerometer_duration
        video_freq = len(forehead) / video_duration

        # upsample the video to the wisci frequency
        new_num_samples = int(len(forehead) * (wisci_freq / video_freq))

        best_corr = -1
        best_lag = 0
        best_n_samples = 0
        corrs = []

        best_signal = None

        for n_samples in tqdm(np.linspace(new_num_samples - 2000, new_num_samples + 2000, 250)):

            resampled_video = resample(forehead, int(n_samples))
            # plot both signals

            corr, lag, corr_array = self.synchronize_by_LED(accelerometer_data, resampled_video)
            corrs.append(corr)
            if corr > best_corr:
                best_corr, best_lag = corr, lag
                best_signal = resampled_video
                best_n_samples = n_samples
                best_corr_array = corr_array
    
        resampled_video = resample(forehead, int(best_n_samples))
        if self.sync_images_folder is not None and original_mp4_basename is not None:

            around = 15*585 # show 15s before (and 15s after) the log
            around = min(around, best_lag) # shorten it from the left side
            around = min(around, len(accelerometer_data) - best_lag, len(best_signal)) # shorten it from the right side
            fig, ax = plt.subplots(figsize=(15, 5), dpi=200)
            ax.plot(np.arange(2*around), accelerometer_data[best_lag-around:best_lag+around], color='red', label='Wisci')
            ax.plot(np.arange(around) + around, best_signal[:around], color='blue', label='From video')
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
            length = min(120*585, len(best_signal[:length]), len(accelerometer_data[best_lag:best_lag+length])) 
            _, ax = plt.subplots(figsize=(15, 5), dpi=200)
            ax.plot(np.arange(length), accelerometer_data[best_lag:best_lag+length], color='red', label='Wisci')
            ax.plot(np.arange(length), best_signal[:length], color='blue', label='From video')
            ax.plot(np.arange(585), np.zeros((585,)), color='black', label='1s stretch', linewidth=2)
            ax.legend()
            plt.savefig(os.path.join(self.sync_images_folder, original_mp4_basename[:-4] + "_two_mins.png"))


            _, ax = plt.subplots(figsize=(15, 5), dpi=200)
            ax.plot(np.arange(len(accelerometer_data)), accelerometer_data, color='red', label='Wisci')
            ax.plot(np.arange(len(best_signal)) + best_lag, best_signal, color='blue', label='From video')
            ax.plot(np.arange(585), np.zeros((585,)), color='black', label='1s stretch', linewidth=2)
            ax.legend()
            plt.savefig(os.path.join(self.sync_images_folder, original_mp4_basename[:-4] + "_whole.png"))

            plt.figure()
            plt.plot(best_corr_array)
            plt.savefig(os.path.join(self.sync_images_folder, original_mp4_basename[:-4] + "_corr.png"))

        peaks, _ = find_peaks(best_corr_array, height=0.5*np.max(best_corr_array), distance=585*2) # TODO: fix distance by the freq
        second_largest_corr_peak = 0
        if len(peaks) < 2:
            second_largest_corr_peak = 0
        else:
            peak_values = np.abs(best_corr_array)[peaks]
            sorted_unique_peak_values = np.sort(np.unique(peak_values))
            second_largest_corr_peak = sorted_unique_peak_values[-2]

        return best_corr, best_lag, best_n_samples, len(peaks), second_largest_corr_peak
    
    
    def sync_with_led(self, mp4_basename, duration_full_path, led_signal_full_path, wisci_file, log=True, output_path_manual=None):

        wisci_data, wisci_duration = get_LED_data(wisci_file)

        video_led = np.load(led_signal_full_path)
        video_duration = np.load(duration_full_path)

        # video_led = video_led - np.min(video_led[1000:]) #np.mean(sig1) # TODO: here was min(), but I changed it because at the beginning the lighting is very low
        # wisci_data = wisci_data - np.min(wisci_data[1000:]) # TODO: here was min(), but I changed it because at the beginning the lighting is very low
        # video_led = video_led / np.max(video_led) - 0.5
        # wisci_data = wisci_data / np.max(wisci_data) - 0.5

        video_led = video_led - np.max(video_led) #np.mean(sig1) # TODO: here was min(), but I changed it because at the beginning the lighting is very low
        wisci_data = wisci_data - np.min(wisci_data) # TODO: here was min(), but I changed it because at the beginning the lighting is very low
        video_led = video_led / (np.max(video_led) - min(video_led[90:-90])) + 0.5
        wisci_data = wisci_data / np.max(wisci_data) - 0.5

        if video_duration > 15: # process videos that are at least 15s long
            best_corr, best_lag, best_n_samples, best_total_peaks, best_second_largest_corr_peak = self.sync_and_optimize_freq(wisci_data, video_led, wisci_duration, video_duration, mp4_basename)
            print(f"Going to update log")
            print(f"File: {mp4_basename}")
            print(f"\tBest correlation: {best_corr}\n\tLag: {best_lag}")
            sync_failed = 0

        else:
            best_corr, best_lag, best_n_samples, best_total_peaks, best_second_largest_corr_peak = -1, -1, -1, -1, -1

            sync_failed = 1
            self.log.update_log(mp4_basename, 'video_duration', video_duration)
            self.log.update_log(mp4_basename, 'sync_error_msg', "Video too short (under 15s).")


        if log:
            self.log.update_log(mp4_basename, 'video_duration', video_duration)
            self.log.update_log(mp4_basename, 'frames', best_n_samples)
            self.log.update_log(mp4_basename, 'path_wisci', wisci_file)
            self.log.update_log(mp4_basename, 'corr', best_corr)
            self.log.update_log(mp4_basename, 'lag', best_lag)
            self.log.update_log(mp4_basename, 'sync_failed', sync_failed)
            self.log.update_log(mp4_basename, 'additional_peaks_per_million', (best_total_peaks-1)/len(wisci_data) * 1000000)
            self.log.update_log(mp4_basename, 'best_second_largest_corr_peak', best_second_largest_corr_peak)
        else:
            # write the logs into a file logs_manual.txt in the output_path_manual (but if the file exists, append to it)
            with open(os.path.join(output_path_manual, 'logs_manual.txt'), 'a') as f:
                f.write(f"Path to the best WiSci file: {wisci_file}\n")
                f.write(f"Normalized correlation: {best_corr}\n")
                f.write(f"Lag: {best_lag}\n")
                f.write(f"Synchronization failed: 0\n")

    def sync_mp4_folder(self):

        processed_files = os.listdir(self.mp4_folder)
        mp4_files_basenames = [os.path.basename(file) for file in processed_files if file.endswith('.mp4')]

        for mp4_basename in mp4_files_basenames:

            try:
                self.log.process_new_file(mp4_basename)
            except FileAlreadySynchronized as e:
                print(e)
                print(f"Skipping {mp4_basename}")
                continue

            print("Processing: ", mp4_basename)
            wisci_file = find_corresponding_wisci(mp4_basename, self.wisci_server_path)
            try:
                self.sync_with_led(
                    mp4_basename,
                    os.path.join(self.mp4_folder, mp4_basename[:-4] + '_duration.npy'),
                    os.path.join(self.led_signals_folder, mp4_basename[:-4] + '_LED_signal.npy'),
                    wisci_file
                )
            except Exception as e:
                print(traceback.format_exc())
                self.log.update_log(mp4_basename, 'sync_error_msg', traceback.format_exc())
                self.log.update_log(mp4_basename, 'sync_failed', 1)
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
    
    def synchronize_by_LED(self, wisci, video, visualize=False):
        normalized_sig1 = wisci
        normalized_sig2 = video
        # sig1 = (sig1 - (np.max(sig1) - np.min(sig1))) / ((np.max(sig1) - np.min(sig1))/2)
        # sig2 = (sig2 - (np.max(sig2) - np.min(sig2))) / ((np.max(sig2) - np.min(sig2))/2)


        correlation = signal.correlate(normalized_sig1, normalized_sig2, mode="valid")
        lags = signal.correlation_lags(normalized_sig1.size, normalized_sig2.size, mode="valid")
        lag = lags[np.argmax(np.abs(correlation))]
        if visualize:
            print(lag)
            plt.plot(np.abs(correlation))
            plt.show()

            plt.plot(np.abs(correlation[np.argmax(np.abs(correlation))-600:np.argmax(np.abs(correlation))+600]))
            plt.title("Peak close-up")
            plt.show()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(len(normalized_sig1)), y=normalized_sig1, mode='lines', name='Wisci', line=dict(color='red')))

            # we computed abs(correlation), to visualize the signals, let's align them
            # (if they were anticorrelated, just put minus before the signal)
            if correlation[np.argmax(np.abs(correlation))] > 0:
                print("pozitivni")
                fig.add_trace(go.Scatter(x=np.arange(len(normalized_sig2)) + lag, y=normalized_sig2, mode='lines', name='From video', line=dict(color='blue')))
            else:
                print("negativni")
                fig.add_trace(go.Scatter(x=np.arange(len(normalized_sig2)) + lag, y=-normalized_sig2, mode='lines', name='From video', line=dict(color='blue')))
            fig.show()

        return np.max(np.abs(correlation)), lag, np.abs(correlation)
    

if __name__=="__main__":

    sigma_video = 10
    sigma_wisci = 200
    visualize = False

    parser = argparse.ArgumentParser(description="Synchronize WiSci and Realsense data using the LED signals.")
    parser.add_argument("mp4_folder", help="Path to the folder containing mp4 files.")
    parser.add_argument("led_signals_folder", help="Path to the folder containing led position .npy files.")
    parser.add_argument("wisci_folder", help="Path to the folder with WISCI .mat files.")
    parser.add_argument("log_table_path", help="Where to log the synchronization.")
    parser.add_argument("sync_images_path", help="Where to store the logs of the synchronization.")
    args = parser.parse_args()

    synchronizer = LedSynchronizer(args.mp4_folder, args.led_signals_folder, args.wisci_folder, args.log_table_path, args.sync_images_path, visualize)
    synchronizer.sync_mp4_folder()
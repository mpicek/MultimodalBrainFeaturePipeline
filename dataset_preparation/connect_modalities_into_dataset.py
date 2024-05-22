import os
import re
import numpy as np
import scipy
import pandas as pd
import sys
import cv2
import pickle
import shutil
from tqdm import tqdm
sys.path.append('../data_synchronization')
from wisci_utils import get_accelerometer_data

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

def concatenate_mat_epochs(folder):
    epochs = [file for file in os.listdir(folder) if file.endswith(".mat") and file.startswith("epoch")]
    sorted_filenames = sorted(epochs, key=extract_number)
    opened_files = [scipy.io.loadmat(os.path.join(folder, file))['epoch_cwt'][0] for file in sorted_filenames]
    concatenated = np.concatenate(opened_files, axis=0)
    return concatenated

def copy_stats_mat_file(original_folder, destination_folder):
    if not os.path.exists(os.path.join(destination_folder, 'stats.mat')):
        shutil.copyfile(os.path.join(original_folder, 'stats.mat'), os.path.join(destination_folder, 'stats.mat'))


def main(
        wisci_location,
        mp4_location, 
        wave_wisci_location, 
        kinematics_folder, 
        output_folder, 
        log_file, 
        led_sync_log_path, 
        acc_sync_log_path
    ):
    ######################################################################################################
    # First, filter out the invalid data files (bad sync, didn't pass the quality test, etc.)
    ######################################################################################################
    led = pd.read_csv(led_sync_log_path)
    acc = pd.read_csv(acc_sync_log_path)
    acc = acc.drop_duplicates()
    led = led.drop_duplicates()

    # join the table based on mp4_name
    df = pd.merge(led, acc, on='mp4_name', how='inner', suffixes=('_led', '_acc'))
    total = df.shape[0]

    # skip files that didn't pass the manual quality test (ONLY WHEN WORKING WITH LED SYNCHRONIZATION)
    df = df.query('passed_quality_test == True')
    # skip files that didn't pass the "metric test" (ONLY WHEN WORKING WITH MOVEMENT (ACC) SYNCHRONIZATION)
    # df = df.query(f'additional_peaks_per_million_acc <= {PEAKS_THRESHOLD}')
    # skip files that failed the synchronization
    df = df.query('sync_failed_led == 0 and sync_failed_acc == 0')
    
    ######################################################################################################
    # Now, loop over THE VIDEOS and create the dataset
    ######################################################################################################
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # There are multiple indexes (time pointers):
        # - _sec: seconds
        # - _585: the frequency of original wisci file ~585.1375Hz
        # - _30: the frequency of the video file ~30Hz
        # - _r: ratio of the length of the wisci file (wisci has length 1)
        #
        # Then we have the so called "increments". It's how big ratio step is done a data point in each file.

        mp4_name = row['mp4_name']

        # skip file if the mp4_name is already in log_file (and check if log_file exists)
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                if mp4_name in f.read():
                    tqdm.write(f"File {mp4_name} already processed. Skipping.")
                    continue

        tqdm.write(f"Processing: {mp4_name}")
        original_wisci_path = row['path_wisci_led']

        basename_wisci = os.path.basename(original_wisci_path)
        last_folder_wisci = os.path.basename(os.path.dirname(original_wisci_path))
        current_wisci_path = os.path.join(wisci_location, last_folder_wisci, basename_wisci)
        current_wave_wisci_path = os.path.join(wave_wisci_location, last_folder_wisci, basename_wisci[:-len('.mat')])
        current_output_folder = os.path.join(output_folder, last_folder_wisci, basename_wisci[:-len('.mat')])
        
        if not os.path.exists(current_output_folder):
            os.makedirs(current_output_folder)

        try:
            copy_stats_mat_file(current_wave_wisci_path, current_output_folder)
        except Exception as e:
            print("Error in file:", current_wave_wisci_path)
            print(e)
            continue

        try:
            kinematics = np.load(os.path.join(kinematics_folder, mp4_name[:-len('.mp4')] + '_processed_kinematics.npy'))
            in_dataset = np.load(os.path.join(kinematics_folder, mp4_name[:-len('.mp4')] + '_processed_in_dataset.npy'))
        except Exception as e:
            print("Error in file:", os.path.join(kinematics_folder, mp4_name[:-len('.mp4')] + '_processed_kinematics.npy'))
            print(e)
            continue

        try:
            acc, wisci_len_sec = get_accelerometer_data(current_wisci_path)
        except Exception as e:
            print("Error in file:", current_wisci_path)
            print(e)
            continue

        wisci_len_585 = len(acc) # Including 6s at the beginning and at the end!!!
        lag_585 = row['lag_led']
        video_len_585 = row['frames_acc']

        # times expressed as ratios of the length of the wisci file
        # wisci length in this ratio is 1 (100%)
        video_len_r = video_len_585 / wisci_len_585
        video_start_r = lag_585 / wisci_len_585
        video_end_r = (lag_585 + video_len_585) / wisci_len_585

        # get video length from opencv
        cap = cv2.VideoCapture(os.path.join(mp4_location, mp4_name))
        video_len_30 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        wave = concatenate_mat_epochs(current_wave_wisci_path)
        zeros_6s = np.zeros((60, wave.shape[1], wave.shape[2]))
        wave = np.concatenate([zeros_6s, wave, zeros_6s], axis=0) # add 6s at the beginning and at the end for easier handling

        video_inc_r = video_len_r / video_len_30
        wisci_inc_r = 1 / wisci_len_585
        wave_inc_r = 1 / (wisci_len_sec * 10) # it's including those 6 seconds at the beginning and at the end


        # Now, loop over the video file and create the datapoints
        video_index = 0
        wave_index = int((video_start_r + video_index * video_inc_r)/ wave_inc_r)

        datapoint = dict()
        datapoint['wisci_path'] = current_wisci_path
        datapoint['wave_wisci_path'] = current_wave_wisci_path
        datapoint['video_path'] = os.path.join(mp4_location, mp4_name)
        datapoint['_video_start'] = video_start_r
        datapoint['_video_end'] = video_end_r
        datapoint['_video_inc'] = video_inc_r
        datapoint['_wisci_inc'] = wisci_inc_r
        datapoint['_wave_inc'] = wave_inc_r
        datapoint['_video_len_30'] = video_len_30
        datapoint['_video_len_585'] = video_len_585
        datapoint['_wisci_len_585'] = wisci_len_585
        datapoint['_wisci_len_sec'] = wisci_len_sec
        datapoint['_lag_585'] = lag_585

        datapoint_index = 0

        while video_index < video_len_30 - 32: # I don't care about those stupid +-1 ones, so to be sure I take 32 frames haha
            wave_index = int((video_start_r + video_index * video_inc_r)/ wave_inc_r)
            
            # The first 6 seconds and last 6 seconds were discarded. So we need to skip them.
            # I'm not sure how the last 6 seconds are handled in the original code from Clinatec, 
            # that's why I discard the whole 7 seconds
            if wave_index < 60:
                video_index += 1
                continue
            if wave_index + 10 > wave.shape[0] - 70: # 10 is the length of the datapoint (the end of the datapoint cannot be in those 6sec)
                break

            # check that the next thirty points in _in_dataset (starting at video_index) are all ones
            # if not, skip this video_index
            if not np.all(in_dataset[video_index:video_index+30] == 1):
                video_index += 1
                continue

            # get the data
            wave_data = wave[wave_index:wave_index+10]
            video_data = kinematics[video_index:video_index+30]

            datapoint['wave_data'] = wave_data
            datapoint['video_data'] = video_data
            datapoint['wave_index'] = wave_index
            datapoint['video_index'] = video_index

            # the file will have 8 digits (so padded with zeros at the beginning)
            output_filename = os.path.join(current_output_folder, str(datapoint_index).zfill(8) + '.pkl')
            with open(output_filename, 'wb') as f:
                pickle.dump(datapoint, f)
            
            datapoint_index += 1
            video_index += 15

        with open(log_file, 'a') as f:
            f.write(row['mp4_name'] + '\n')


PEAKS_THRESHOLD = 5
wisci_location = '/media/mpicek/T7/martin/new_WISCI/'
mp4_location = '/media/mpicek/T7/martin/data_final/mp4/'
wave_wisci_location = '/media/mpicek/T7/icare/PROCESSED_DATA_UP2001/all/'
kinematics_folder = '/media/mpicek/T7/martin/data_final/kinematics/'
output_folder = '/media/mpicek/T7/martin/data_final/dataset/'
log_file = '/media/mpicek/T7/martin/data_final/files_used_for_dataset.txt'
led_sync_log_path = '/media/mpicek/T7/martin/data_final/log_sync_led.csv'
acc_sync_log_path = '/media/mpicek/T7/martin/data_final/log_sync_acc.csv'

if __name__ == '__main__':
    # pass everything to main
    main(wisci_location, 
        mp4_location, 
        wave_wisci_location, 
        kinematics_folder, 
        output_folder, 
        log_file, 
        led_sync_log_path, 
        acc_sync_log_path)
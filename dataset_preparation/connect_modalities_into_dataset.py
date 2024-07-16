import os
import re
import numpy as np
import scipy
import pandas as pd
import sys
import cv2
import pickle
import shutil
import scipy.signal
from tqdm import tqdm
sys.path.append('../data_synchronization')
from wisci_utils import get_accelerometer_data
from synchronization_utils import smooth


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
        dino_folder,
        in_dataset_dino_folder,
        output_folder, 
        log_file, 
        led_sync_log_path, 
        acc_sync_log_path,
        kinematics_folder,
        in_dataset_kinematics_folder
    ):
    """
    Connects all modalities into one dataset:
      - wavelet transform of WIMAGINE data (10Hz)
      - DINO features (30Hz)
      - accelerometer data (30Hz) - the raw 585Hz data is smoothed and resampled to 30Hz
    """
    
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
        # There are multiple indices (time pointers):
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
        # the segmented video has to exist
        # (because during the manual data quality check, I delete the videos that are not good enough)
        if not os.path.join(in_dataset_dino_folder, mp4_name):
            print(f"The segmented video {mp4_name} doesn't exist in the in_dataset_dino_folder folder. Skipping.")
            continue

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
            dino_features = np.load(os.path.join(dino_folder, mp4_name[:-len('.mp4')] + '_features.npy'))
            in_dataset_dino = np.load(os.path.join(in_dataset_dino_folder, mp4_name[:-len('.mp4')] + '_segmented_pose_temp_in_dataset.npy'))
        except Exception as e:
            print("Error in file:", os.path.join(dino_folder, mp4_name[:-len('.mp4')] + '_features.npy'))
            print(e)
            continue

        try:
            acc, wisci_len_sec = get_accelerometer_data(current_wisci_path)
            acc = acc - np.mean(acc, axis=0)
            acc[:, 0] = smooth(acc[:, 0], 10)
            acc[:, 1] = smooth(acc[:, 1], 10)
            acc[:, 2] = smooth(acc[:, 2], 10)
            # later we also resample the signal to 30Hz (always the one second that we use)
        except Exception as e:
            print("Error in file:", current_wisci_path)
            print(e)
            continue

        include_kinematics = False
        try:
            kinematics = np.load(os.path.join(kinematics_folder, mp4_name[:-len('.mp4')] + '_processed_2d_kinematics.npy'))
            # in dataset is not necessary because it can be 'read' directly from kinematics. If it's nan, then it's not in the dataset.
            # in_dataset_kinematics = np.load(os.path.join(kinematics_folder, mp4_name[:-len('.mp4')] + '_processed_in_dataset.npy'))
            include_kinematics = True
        except Exception as e:
            print("Problem with kinematics. The file is NOT skipped, but the kinematics will not be in the generated datapoints.")
        

        # make sure that kinematics is the same length as the video
        if include_kinematics:
            try:
                assert kinematics.shape[0] == in_dataset_dino.shape[0]
            except Exception as e:
                print("Kinematics and in_dataset_dino have different lengths. Skipping.")
                continue

        wisci_len_585 = len(acc) # Including 6s at the beginning and at the end!!!
        lag_585 = row['lag_led']
        video_len_585 = row['frames_acc'] # the length of the video in wisci freq after correct resampling

        # times expressed as ratios of the length of the wisci file
        # wisci length in this ratio is 1 (100%)
        video_len_r = video_len_585 / wisci_len_585
        video_start_r = lag_585 / wisci_len_585
        video_end_r = (lag_585 + video_len_585) / wisci_len_585

        # get video length from opencv
        cap = cv2.VideoCapture(os.path.join(mp4_location, mp4_name))
        video_len_30 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        wave = concatenate_mat_epochs(current_wave_wisci_path) # connect features into one big array (spectrogram)
        zeros_6s = np.zeros((60, wave.shape[1], wave.shape[2]))
        wave = np.concatenate([zeros_6s, wave, zeros_6s], axis=0) # add 6s at the beginning and at the end for easier handling

        video_inc_r = video_len_r / video_len_30 # increment by one frame
        wisci_inc_r = 1 / wisci_len_585          # increment by one WIMAGINE tick (585.1375Hz)
        wave_inc_r = 1 / (wisci_len_sec * 10)    # increment by 0.1s, it's including those 6 seconds at the beginning and at the end!


        # Now, loop over the video file and create the datapoints

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

        video_index = 0
        datapoint_index = 0

        while video_index < video_len_30 - 32: # I don't care about those stupid +-1 ones, so to be sure I take 32 frames haha
            wave_index = int((video_start_r + video_index * video_inc_r)/ wave_inc_r)
            acc_index = int((video_start_r + video_index * video_inc_r)/ wisci_inc_r)

            
            # The first 6 seconds and last 6 seconds were discarded. So we need to skip them.
            # I'm not sure how the last 6 seconds are handled in the original code from Clinatec, 
            # that's why I discard the whole 7 seconds
            if wave_index < 60:
                video_index += 1
                continue
            if wave_index + 10 > wave.shape[0] - 70: # 10 is the length of the datapoint (the end of the datapoint cannot be in those 6sec)
                break

            # check that the next thirty points in _in_dataset_dino (starting at video_index) are all ones
            # if not, skip this video_index
            if not np.all(in_dataset_dino[video_index:video_index+30] == 1):
                video_index += 1
                continue

            # get the data
            wave_data = wave[wave_index:wave_index+10]
            current_dino_features = dino_features[video_index:video_index+30]
            current_accelerometer = acc[acc_index:acc_index + 585]
            # also smooth the raw accelerometer data
            current_accelerometer = scipy.signal.resample(current_accelerometer, 30, axis=0)
            current_kinematics = kinematics[video_index:video_index+30] if include_kinematics else None


            datapoint['wave_data'] = wave_data
            datapoint['dino_features'] = current_dino_features
            datapoint['acc_data'] = current_accelerometer
            datapoint['kinematics'] = current_kinematics
            datapoint['wave_index'] = wave_index
            datapoint['video_index'] = video_index
            datapoint['acc_index'] = acc_index
            datapoint['kinematics_index'] = video_index

            # the file will have 8 digits (so padded with zeros at the beginning)
            output_folder_for_current_video = os.path.join(current_output_folder, mp4_name[:-len('.mp4')])
            if not os.path.exists(output_folder_for_current_video):
                os.makedirs(output_folder_for_current_video)
            output_filename = os.path.join(output_folder_for_current_video, str(datapoint_index).zfill(8) + '.pkl')
            with open(output_filename, 'wb') as f:
                pickle.dump(datapoint, f)
            
            datapoint_index += 1
            video_index += 15

        with open(log_file, 'a') as f:
            f.write(row['mp4_name'] + '\n')


PEAKS_THRESHOLD = 5
wisci_location = '/media/cyberspace007/T7/martin/WISCI/'
mp4_location = '/media/cyberspace007/T7/martin/data/mp4/'
wave_wisci_location = '/media/cyberspace007/T7/icare/PROCESSED_DATA_UP2001/all/'
dino_folder = '/media/cyberspace007/T7/martin/data/dino_features_0_7'
in_dataset_dino_folder = '/media/cyberspace007/T7/martin/data/pose_segmentation_0_7_filtered'
# output_folder = '/media/cyberspace007/T7/martin/data/dataset_0_7/'
output_folder = '/media/cyberspace007/T7/martin/data/dataset_with_kinematics_0_7/'
# log_file = '/media/cyberspace007/T7/martin/data/files_used_for_dataset_0_7.txt'
log_file = '/media/cyberspace007/T7/martin/data/files_used_for_dataset_with_kinematics_0_7.txt'
led_sync_log_path = '/media/cyberspace007/T7/martin/data/log_sync_led.csv'
acc_sync_log_path = '/media/cyberspace007/T7/martin/data/log_sync_acc.csv'
kinematics_folder = '/media/cyberspace007/T7/martin/data/kinematics'
in_dataset_kinematics_folder = kinematics_folder

if __name__ == '__main__':
    # pass everything to main
    main(wisci_location, 
        mp4_location, 
        wave_wisci_location, 
        dino_folder,
        in_dataset_dino_folder,
        output_folder, 
        log_file, 
        led_sync_log_path,
        acc_sync_log_path,
        kinematics_folder,
        in_dataset_kinematics_folder)
import pandas as pd
import scipy.io
import numpy as np
from synchronization_utils import find_data_in_wisci
import os
import re
from datetime import datetime

class GettingAccelerometerDataFailed(Exception):
    pass

def get_accelerometer_data(mat_file):
    """
    Extracts and processes accelerometer data from a MATLAB file, 
    interpolating missing values and subsampling to match a specified frequency.

    This function performs the following steps:
    1. Loads accelerometer data from a specified MATLAB file.
    2. Interpolates missing values (NaNs) linearly to fill gaps in the data.
    3. Resamples the data to achieve the sampling rate of 30Hz.

    Parameters:
    - mat_file (str): The path to the MATLAB (.mat) file containing accelerometer data (aka wisci file).

    Returns:
    - numpy.ndarray: A 2D array of shape (n, 3), where 'n' is the number of subsampled data points in 30Hz.
      Each row corresponds to a data point, 
      with columns representing the interpolated and subsampled x, y, and z accelerometer readings, respectively.

    Note:
    - The function assumes the accelerometer data is stored in the MATLAB file under the key 'STREAMS' > 'Signal_stream_1'.
    - It is also assumed that the original sampling rate of the accelerometer data is 585.1375 Hz.
    - The interpolation and subsampling process is designed to synchronize accelerometer data with video data recorded at 30 Hz,
      by effectively reducing the accelerometer data's sampling rate to match the video frame rate.
    """
    
    mat = scipy.io.loadmat(mat_file)

    acc_stream, _ = find_data_in_wisci(mat, 'raccx')

    data_acc = mat['STREAMS'][acc_stream][0][0]['data'][0][0][:,:,0].T
    df = pd.DataFrame(data_acc, columns=['x', 'y', 'z'])
    oversampled = df.interpolate(method='linear', axis=0).bfill()

    interpolated_data = np.column_stack((oversampled['x'], oversampled['y'], oversampled['z']))

    duration = find_wisci_duration(mat, acc_stream)

    return interpolated_data, duration


def get_LED_data(mat_file):
    mat = scipy.io.loadmat(mat_file)

    led_stream, led_index = find_data_in_wisci(mat, 'tr7', 'tr2')
    led_data = mat['STREAMS'][led_stream][0][0]['data'][0][0][:,:,0][led_index].T

    df = pd.DataFrame(led_data, columns=['led'])
    # we remove the nans by interpolating -> the freq is the same as the highest freq of wisci
    original_freq = df.interpolate(method='linear', axis=0).bfill()

    duration = find_wisci_duration(mat, led_stream)
    
    return original_freq['led'], duration


def find_wisci_duration(mat, stream_name):
    """
    Returns in seconds!!
    """
    start = float(mat['STREAMS'][stream_name][0][0]['t_start'][0][0][0][0])
    stop = float(mat['STREAMS'][stream_name][0][0]['t_stop'][0][0][0][0])
    return stop - start

def find_corresponding_wisci(mp4_filename, wisci_path):
    # Extract datetime from mp4 filename
    mp4_datetime = re.search(r'(\d{2}_\d{2}_\d{4}_\d{4}_\d{2})', mp4_filename).group(0)
    mp4_datetime = datetime.strptime(mp4_datetime, '%d_%m_%Y_%H%M_%S')
    
    # Find .mat files in the directory and subdirectories
    mat_files = []
    for root, dirs, files in os.walk(wisci_path):
        for file in files:
            if file.endswith('.mat'):
                mat_files.append(os.path.join(root, file))
    
    # Function to extract datetime from .mat filename
    def extract_datetime(mat_filename):
        mat_datetime = re.search(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2})', mat_filename).group(0)
        return datetime.strptime(mat_datetime, '%Y_%m_%d_%H_%M')

    # Filter out invalid .mat files that don't contain datetime    
    valid_mat_files = []
    for mat_file in mat_files:
        try:
            valid_mat_files.append((mat_file, extract_datetime(mat_file)))
        except:
            pass

    valid_mat_files.sort(key=lambda x: x[1])  # Sort by datetime

    # Find the closest lower time .mat file
    closest_mat_file = None
    for mat_file, mat_datetime in valid_mat_files:
        if mat_datetime <= mp4_datetime:
            closest_mat_file = mat_file
        else:
            break
    
    return closest_mat_file
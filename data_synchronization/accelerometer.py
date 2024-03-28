# import pandas as pd
# import scipy.io
# import numpy as np
# from scipy.signal import resample
# from synchronization_utils import find_data_in_wisci

# class GettingAccelerometerDataFailed(Exception):
#     pass

# def get_accelerometer_data(mat_file):
#     """
#     Extracts and processes accelerometer data from a MATLAB file, 
#     interpolating missing values and subsampling to match a specified frequency.

#     This function performs the following steps:
#     1. Loads accelerometer data from a specified MATLAB file.
#     2. Interpolates missing values (NaNs) linearly to fill gaps in the data.
#     3. Resamples the data to achieve the sampling rate of 30Hz.

#     Parameters:
#     - mat_file (str): The path to the MATLAB (.mat) file containing accelerometer data (aka wisci file).

#     Returns:
#     - numpy.ndarray: A 2D array of shape (n, 3), where 'n' is the number of subsampled data points in 30Hz.
#       Each row corresponds to a data point, 
#       with columns representing the interpolated and subsampled x, y, and z accelerometer readings, respectively.

#     Note:
#     - The function assumes the accelerometer data is stored in the MATLAB file under the key 'STREAMS' > 'Signal_stream_1'.
#     - It is also assumed that the original sampling rate of the accelerometer data is 585.1375 Hz.
#     - The interpolation and subsampling process is designed to synchronize accelerometer data with video data recorded at 30 Hz,
#       by effectively reducing the accelerometer data's sampling rate to match the video frame rate.
#     """
    
#     mat = scipy.io.loadmat(mat_file)

#     acc_stream, _ = find_data_in_wisci(mat, 'raccx')

#     data_acc = mat['STREAMS'][acc_stream][0][0]['data'][0][0][:,:,0].T
#     df = pd.DataFrame(data_acc, columns=['x', 'y', 'z'])
#     oversampled = df.interpolate(method='linear', axis=0).bfill()

#     original_freq = 585.1375
#     desired_rate = 30
#     print(len(oversampled))

#     # Calculate the duration of your signal
#     num_samples = len(oversampled)
#     duration = num_samples / original_freq

#     # Calculate the new number of samples required for the desired rate
#     new_num_samples = int(np.round(duration * desired_rate))

#     interpolated_x = resample(oversampled['x'], new_num_samples)
#     interpolated_y = resample(oversampled['y'], new_num_samples)
#     interpolated_z = resample(oversampled['z'], new_num_samples)

#     interpolated_data = np.column_stack((interpolated_x, interpolated_y, interpolated_z))

#     return interpolated_data[::6]

import pandas as pd
import scipy.io
import numpy as np
from scipy.signal import resample
from synchronization_utils import find_data_in_wisci

class GettingAccelerometerDataFailed(Exception):
    pass

# def get_accelerometer_data(mat_file):
#     """
#     Extracts and processes accelerometer data from a MATLAB file, 
#     interpolating missing values and subsampling to match a specified frequency.

#     This function performs the following steps:
#     1. Loads accelerometer data from a specified MATLAB file.
#     2. Interpolates missing values (NaNs) linearly to fill gaps in the data.
#     3. Resamples the data to achieve the sampling rate of 30Hz.

#     Parameters:
#     - mat_file (str): The path to the MATLAB (.mat) file containing accelerometer data (aka wisci file).

#     Returns:
#     - numpy.ndarray: A 2D array of shape (n, 3), where 'n' is the number of subsampled data points in 30Hz.
#       Each row corresponds to a data point, 
#       with columns representing the interpolated and subsampled x, y, and z accelerometer readings, respectively.

#     Note:
#     - The function assumes the accelerometer data is stored in the MATLAB file under the key 'STREAMS' > 'Signal_stream_1'.
#     - It is also assumed that the original sampling rate of the accelerometer data is 585.1375 Hz.
#     - The interpolation and subsampling process is designed to synchronize accelerometer data with video data recorded at 30 Hz,
#       by effectively reducing the accelerometer data's sampling rate to match the video frame rate.
#     """
    
#     mat = scipy.io.loadmat(mat_file)

#     acc_stream, _ = find_data_in_wisci(mat, 'raccx')

#     data_acc = mat['STREAMS'][acc_stream][0][0]['data'][0][0][:,:,0].T
#     df = pd.DataFrame(data_acc, columns=['x', 'y', 'z'])
#     oversampled = df.interpolate(method='linear', axis=0).bfill()

#     original_freq = 585.1375
#     desired_rate = 30

#     # Calculate the duration of your signal
#     num_samples = len(oversampled)
#     duration = num_samples / original_freq

#     # Calculate the new number of samples required for the desired rate
#     new_num_samples = int(np.round(duration * desired_rate))

#     interpolated_x = resample(oversampled['x'], new_num_samples)
#     interpolated_y = resample(oversampled['y'], new_num_samples)
#     interpolated_z = resample(oversampled['z'], new_num_samples)

#     interpolated_data = np.column_stack((interpolated_x, interpolated_y, interpolated_z))

#     return interpolated_data

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
    
    return original_freq, duration


def find_wisci_duration(mat, stream_name):
    """
    Returns in seconds!!
    """
    start = float(mat['STREAMS'][stream_name][0][0]['t_start'][0][0][0][0])
    stop = float(mat['STREAMS'][stream_name][0][0]['t_stop'][0][0][0][0])
    return stop - start
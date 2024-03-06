import pandas as pd
import scipy.io
import numpy as np

def get_accelerometer_data(mat_file):
    """
    Extracts and processes accelerometer data from a MATLAB file, 
    interpolating missing values and subsampling to match a specified frequency.

    This function performs the following steps:
    1. Loads accelerometer data from a specified MATLAB file.
    2. Interpolates missing values (NaNs) linearly to fill gaps in the data.
    3. Subsamples the interpolated data to match the target frequency of 30 Hz, 
       which involves selecting every 19.5th data point to align with video frame rate for synchronization purposes.

    Parameters:
    - mat_file (str): The path to the MATLAB (.mat) file containing accelerometer data.

    Returns:
    - numpy.ndarray: A 2D array of shape (n, 3), where 'n' is the number of subsampled data points in 30Hz.
      Each row corresponds to a data point, 
      with columns representing the interpolated and subsampled x, y, and z accelerometer readings, respectively.

    Note:
    - The function assumes the accelerometer data is stored in the MATLAB file under the key 'STREAMS' > 'Signal_stream_1'.
    - It is also assumed that the original sampling rate of the accelerometer data is ~585Hz.
    - The interpolation and subsampling process is designed to synchronize accelerometer data with video data recorded at 30 Hz,
      by effectively reducing the accelerometer data's sampling rate to match the video frame rate.
    """
    
    mat = scipy.io.loadmat(mat_file)
    data_acc = mat['STREAMS']['Signal_stream_1'][0][0][0][0][0][:,:,0].T
    df = pd.DataFrame(data_acc, columns=['x', 'y', 'z'])
    oversampled = df.interpolate(method='linear', axis=0).bfill()

    # at this point, we have interpolated data instead of nans (it is "oversampled")
    # now, we need to subsample it to the same frequency as the video (30 Hz) -> 585/30 = 19.5
    # so we take every 19.5th measurement

    fractional_indices = np.arange(0, len(oversampled), 19.5)

    interpolated_data = np.zeros((len(fractional_indices), 3)) # here the data will be stored

    # Original indices (from 0 to length of your array)
    original_indices = np.arange(len(oversampled))

    for i, coordinate in enumerate(['x', 'y', 'z']):
        column_data = oversampled[coordinate] # because it's a pandas dataframe
        interpolated_data[:, i] = np.interp(fractional_indices, original_indices, column_data)    

    return interpolated_data


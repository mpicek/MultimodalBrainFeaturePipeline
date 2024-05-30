import sys
sys.path.append('../data_synchronization')
import numpy as np
import matplotlib.pyplot as plt
import os
from synchronization_utils import smooth
import plotly.graph_objects as go
import argparse
from tqdm import tqdm

################################################################################
# Preprocess the kinematics signals
# Selects only the useful joints for the UP2 dataset
# Detects jumps caused by physios' hands
# Smooths the signals
# Computes the gradient of the smoothed signals
# Visualizes the processed signals (optional)
# Saves the processed signals (with _processed_kinematics.npy and _processed_in_dataset.npy suffixes)
################################################################################

get_joint_name = {
    0 : "nose",
    1 : "left eye (inner)",
    2 : "left eye",
    3 : "left eye (outer)",
    4 : "right eye (inner)",
    5 : "right eye",
    6 : "right eye (outer)",
    7 : "left ear",
    8 : "right ear",
    9 : "mouth (left)",
    10: "mouth (right)",
    11: "left shoulder",
    12: "right shoulder",
    13: "left elbow",
    14: "right elbow",
    15: "left wrist",
    16: "right wrist",
    17: "left pinky",
    18: "right pinky",
    19: "left index",
    20: "right index",
    21: "left thumb",
    22: "right thumb",
    23: "left hip",
    24: "right hip",
    25: "left knee",
    26: "right knee",
    27: "left ankle",
    28: "right ankle",
    29: "left heel",
    30: "right heel",
    31: "left foot index",
    32: "right foot index",
}

get_joint_index = {v: k for k, v in get_joint_name.items()}

get_joint_name_UP2 = {
    0 : "nose",
    1 : "left elbow",
    2 : "right elbow",
    3 : "left wrist",
    4 : "right wrist",
    5 : "left pinky",
    6 : "right pinky",
}

get_joint_index_UP2 = {v: k for k, v in get_joint_name_UP2.items()}

def compute_average_arm_length_in_pixels(joints):
    """
    Computes the average arm length using the joints array.
    
    Args:
        joints (np.ndarray): Shape (frames, joints, 4) .. the original joints from mediapipe
        
    Returns:
        float: The average arm length in pixels.
    """
    left_arm = joints[:, get_joint_index["left shoulder"], :2] - joints[:, get_joint_index["left elbow"], :2]
    right_arm = joints[:, get_joint_index["right shoulder"], :2] - joints[:, get_joint_index["right elbow"], :2]

    # ignore nan values during computation
    avg_left_arm_length = np.nanmean(np.linalg.norm(left_arm, axis=1))
    avg_right_arm_length = np.nanmean(np.linalg.norm(right_arm, axis=1))

    avg_arm_length = (avg_left_arm_length + avg_right_arm_length) / 2
    return avg_arm_length

def pixels2cm_using_armlength(joints, arm_length_in_cm, avg_arm_length_in_pixels=None):
    """
    Converts pixel values to cm using the average arm length.
    Detects patient's arms (elbow to shoulder) and uses the average length of both arms to convert the signal to cm.

    Args:
        joints (np.ndarray): Shape (frames, joints, 4) .. the original joints from mediapipe
        arm_length_in_cm (float): True patient's arm length in cm.
        avg_arm_length_in_pixels (float): Precomputed average arm length in pixels (optional).

    Returns:
        np.ndarray: The signal but not in pixels but in cm.
    """
    if avg_arm_length_in_pixels is None:
        avg_arm_length_in_pixels = compute_average_arm_length_in_pixels(joints)
    
    joints_cm = joints.copy()
    joints_cm[:, :, :2] /= avg_arm_length_in_pixels
    joints_cm[:, :, :2] *= arm_length_in_cm

    return joints_cm

def cm2pixels_using_armlength(joints_cm, arm_length_in_cm, avg_arm_length_in_pixels):
    """
    Converts cm values back to pixels using the average arm length.
    
    Args:
        joints_cm (np.ndarray): Shape (frames, joints, 4) .. the joints in cm
        arm_length_in_cm (float): True patient's arm length in cm.
        avg_arm_length_in_pixels (float): The average arm length in pixels.
    
    Returns:
        np.ndarray: The signal but not in cm but in pixels.
    """
    joints_pixels = joints_cm.copy()
    joints_pixels[:, :, :2] /= arm_length_in_cm
    joints_pixels[:, :, :2] *= avg_arm_length_in_pixels

    return joints_pixels

def get_useful_joints_UP2(joints):
    """
    Returns only the useful joints for the UP2 dataset.
    Args:
        joints (np.ndarray): Shape (frames, joints, 4) .. the original joints from mediapipe
            need to be used
    Returns:
        np.ndarray: Shape (frames, useful_joints, 4) .. the useful joints for the UP2 dataset
    """
    useful_points = [
        get_joint_index["nose"],
        get_joint_index["left elbow"],
        get_joint_index["right elbow"],
        get_joint_index["left wrist"],
        get_joint_index["right wrist"],
        get_joint_index["left thumb"],
        get_joint_index["right thumb"],
        get_joint_index["left shoulder"],
        get_joint_index["right shoulder"],
    ]
    return joints[:, useful_points, :]

def smooth_joints_UP2(joints, SIGMA):
    """
    Smooths the joints for the UP2 dataset.
    Args:
        joints (np.ndarray): Shape (frames, joints, 4)
        SIGMA (int): The SIGMA value for the gaussian filter
    Returns:
        np.ndarray: Shape (frames, joints, 4) .. the smoothed joints
    """
    for index in range(joints.shape[1]):
        joints[:, index, 0] = smooth(joints[:, index, 0], SIGMA)
        joints[:, index, 1] = smooth(joints[:, index, 1], SIGMA)
    return joints

def plot(joints, indices=None, names=[], x_max=None, y_max=None):
    """Plots the joints next to each other.

    Args:
        joints (np.ndarray): Shape (frames, joints, 2) or (frames, joints)
        indices (np.ndarray): Shape (num_indices,) .. the indices of the joints to plot (if None, all joints are plotted)
        names (list, optional): Descriptions for the . Defaults to [].
        x_max (_type_, optional): _description_. Defaults to None.
        y_max (_type_, optional): _description_. Defaults to None.
    """
    data = []
    if indices is None:
        indices = range(joints.shape[1])
    for index in indices:
        if len(joints.shape) == 3: # if it has the third dimension with x and y
            data.append(go.Scatter(y=joints[:, index, 0], name=get_joint_name[index] + "_x"))
            data.append(go.Scatter(y=joints[:, index, 1], name=get_joint_name[index] + "_y"))
        else: # if it's just one signal for each joint
            data.append(go.Scatter(y=joints[:, index], name=names[index]))

    fig = go.Figure(data=data)
    # change the labels on x axis (divide them by 30 to get seconds from FPS)
    fig.update_xaxes(tickvals=np.arange(0, joints.shape[0], 30), ticktext=np.arange(0, joints.shape[0] // 30))
    if x_max is not None:
        fig.update_xaxes(range=[0, x_max])
    if y_max is not None:
        fig.update_yaxes(range=[-y_max, y_max])
    fig.show()

def plot_one_signal(signal):
    """Plots one signal."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=signal))
    fig.show()

def detect_jumps(signal, threshold):
    """
    Detects jumps in the signal FOR EACH JOINT INDIVIDUALLY.
    Args:
        signal (np.ndarray): The NOT SMOOTHED POSITION (not velocity) signal to detect jumps in. Shape (frames, joints, 2)
        threshold (float): The threshold for the jump detection.
    Returns:
        np.ndarray: The signal with jumps detected. Shape (frames,)
    """

    jumps = np.zeros((signal.shape[0], signal.shape[1]))
    for i in range(1, signal.shape[0]):
        # jump either in x or y coordinate for each joint
        for j in range(signal.shape[1]):
            if np.abs(signal[i][j][0] - signal[i - 1][j][0]) > threshold or np.abs(signal[i][j][1] - signal[i - 1][j][1]) > threshold:
                jumps[i][j] = 1
    return jumps

def set_jumps_to_nan(joints, jumps):
    """
    Sets NaNs in joints for indices where jumps are detected.
    
    Args:
        joints (np.ndarray): The joint positions. Shape (frames, joints, 2)
        jumps (np.ndarray): The detected jumps. Shape (frames, joints)
    
    Returns:
        np.ndarray: The joint positions with NaNs set for detected jumps.
    """
    for i in range(jumps.shape[0]):
        for j in range(jumps.shape[1]):
            if jumps[i][j] == 1:
                joints[i][j] = [np.nan, np.nan, np.nan, np.nan]
    return joints

def set_not_visible_joints_to_nan(joints):
    """
    Sets NaNs in joints for indices where joints are not visible.
    
    Args:
        joints (np.ndarray): The joint positions. Shape (frames, joints, 2)
    
    Returns:
        np.ndarray: The joint positions with NaNs set for non-visible joints.
    """

    for i in range(joints.shape[0]):
        for j in range(joints.shape[1]):
            if joints[i, j, 3] < VISIBILIT_THRESHOLD:
                joints[i][j] = [np.nan, np.nan, np.nan, np.nan]
    return joints

def process_kinematics_file(joints):

    # we need to normalize the data (to the arm length) because we need to detect jumps with the same threshold
    # this threshold has to work over all the videos, that's why we need to normalize the data
    avg_arm_length_in_pixels = compute_average_arm_length_in_pixels(joints)
    joints = pixels2cm_using_armlength(joints, ARM_LENGTH_IN_CM, avg_arm_length_in_pixels)
    joints = get_useful_joints_UP2(joints)
    jumps = detect_jumps(joints, JUMP_THRESHOLD_CM)
    joints = set_jumps_to_nan(joints, jumps)
    joints = cm2pixels_using_armlength(joints, ARM_LENGTH_IN_CM, avg_arm_length_in_pixels)
    
    joints = set_not_visible_joints_to_nan(joints)

    # 1. smooth the signal
    for joint in range(joints.shape[1]):
        joints[:, joint, 0] = smooth(joints[:, joint, 0], SIGMA)
        joints[:, joint, 1] = smooth(joints[:, joint, 1], SIGMA)
        joints[:, joint, 2] = smooth(joints[:, joint, 2], SIGMA)
    

    # 2. smooth the signal
    # for joint in range(gradient.shape[1]):
    #     gradient[:, joint, 0] = smooth(gradient[:, joint, 0], SIGMA)
    #     gradient[:, joint, 1] = smooth(gradient[:, joint, 1], SIGMA)
    #     gradient[:, joint, 2] = smooth(gradient[:, joint, 2], SIGMA)
    

    # 3. compute the gradient for each joint separately and smooth it
    # final_gradient = np.zeros((joints.shape[0], joints.shape[1]))
    # for joint in range(gradient.shape[1]):
    #     gradient[:, joint, 0] = smooth(np.gradient(gradient[:, joint, 0]), SIGMA)
    #     gradient[:, joint, 1] = smooth(np.gradient(gradient[:, joint, 1]), SIGMA)
    #     # 4. decrease the movement of z coordinate by a factor (because it's very noisy)
    #     gradient[:, joint, 2] = smooth(np.gradient(gradient[:, joint, 2] * Z_FACTOR), SIGMA)

        # 5. compute the size (norm) of the gradient
        # if USE_Z_COORDINATE:
        #     final_gradient[:, joint] = np.linalg.norm(gradient[:, joint, :3], axis=1)
        # else:
        #     final_gradient[:, joint] = np.linalg.norm(gradient[:, joint, :2], axis=1)
    


    # if VISUALIZE:
    #     jumps *= 80
    #     non_smoothed_gradient = joints.copy()
    #     print(non_smoothed_gradient.shape)
    #     for i in range(jumps.shape[1]):
    #         jumps[:, i] += i * 5
    #     final_non_smoothed_gradient = np.zeros((joints.shape[0], joints.shape[1]))
    #     for joint in range(non_smoothed_gradient.shape[1]):
    #         non_smoothed_gradient[:, joint, 0] = np.gradient(non_smoothed_gradient[:, joint, 0])
    #         non_smoothed_gradient[:, joint, 1] = np.gradient(non_smoothed_gradient[:, joint, 1])
    #         final_non_smoothed_gradient[:, joint] = np.linalg.norm(non_smoothed_gradient[:, joint, :2], axis=1)


    #     # we visualize the smoothed, not smoothed signals and the big jumps in coordinates
    #     nose = np.concatenate((final_gradient[:, :7], final_non_smoothed_gradient[:, :7], jumps[:, :7]), axis=1)
    #     print(nose.shape)
    #     plot(nose, np.arange(21), names = [
    #         'nose',
    #         'left elbow',
    #         'right elbow',
    #         'left wrist',
    #         'right wrist',
    #         'left thumb',
    #         'right thumb',
    #         'nose',
    #         'left elbow original',
    #         'right elbow original',
    #         'left wrist original',
    #         'right wrist original',
    #         'left thumb original',
    #         'right thumb original',
    #         'jumps nose',
    #         'jumps left elbow',
    #         'jumps right elbow',
    #         'jumps left wrist',
    #         'jumps right wrist',
    #         'jumps left thumb',
    #         'jumps right thumb',
    #     ])
    
    # new_in_dataset_signal = []
    # for frame in final_gradient:
    #     if np.all(np.isnan(frame)):
    #         new_in_dataset_signal.append(0)
    #     else:
    #         new_in_dataset_signal.append(1)

    new_in_dataset_signal = []
    for frame in joints:
        if np.all(np.isnan(frame)):
            new_in_dataset_signal.append(0)
        else:
            new_in_dataset_signal.append(1)
    
    return joints, new_in_dataset_signal

def process_folder(input_folder, output_folder):
    """
    Processes the folder with kinematics files.
    Args:
        folder (str): The folder with kinematics files.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in tqdm(os.listdir(input_folder)):
        if file.endswith("_kinematics.npy") and not file.endswith("_processed_2d_kinematics.npy") and not file.endswith("_processed_kinematics.npy"):
            print(file)
            # if the _processed_kinematics.npy file already exists, skip it
            if os.path.exists(os.path.join(output_folder, file.replace("_kinematics.npy", "_processed_2d_kinematics.npy"))):
                tqdm.write(f"File {file} already processed. Skipping.")
                continue
            joints = np.load(os.path.join(input_folder, file))
            processed_joints, in_dataset = process_kinematics_file(joints)
            np.save(os.path.join(output_folder, file.replace("_kinematics.npy", "_processed_2d_kinematics.npy")), processed_joints)
            np.save(os.path.join(output_folder, file.replace("_kinematics.npy", "_processed_2d_in_dataset.npy")), in_dataset)

SIGMA = 1 # shortens the signal by 8 frames (by two applications of smoothing)
ARM_LENGTH_IN_CM = 30
JUMP_THRESHOLD_CM = 10 # without prior smoothing! So it's higher because there can be noise
Z_FACTOR = 1/10 # decrease the movement of z coordinate by this factor
USE_Z_COORDINATE = False
VISUALIZE = False
VISIBILIT_THRESHOLD = 0.75

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the kinematics signals.")
    parser.add_argument("extraction_folder", help="Path to the folder containing kinematics .npy files.")
    parser.add_argument("output_folder", help="Path to the output folder to save the processed kinematics signals with processed_in_dataset.npy.")
    args = parser.parse_args()

    process_folder(args.extraction_folder, args.output_folder)

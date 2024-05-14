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

def pixels2cm_using_armlength(joints, arm_length_in_cm):
    """
    Converts pixel values to cm using the average arm length.
    Detects patient's arms (elbow to shoulder) and uses the average length of both arms to convert the signal to cm.

    Args:
        joints (np.ndarray): Shape (frames, joints, 2) .. the original joints from mediapipe
            need to be used
        arm_length_in_cm (float): True patient's arm length in cm.

    Returns:
        np.ndarray: The signal but not in pixels but in cm.
    """
    left_arm = joints[:, get_joint_index["left shoulder"], :] - joints[:, get_joint_index["left elbow"], :]
    right_arm = joints[:, get_joint_index["right shoulder"], :] - joints[:, get_joint_index["right elbow"], :]

    # ignore nan values during computation
    avg_left_arm_length = np.nanmean(np.linalg.norm(left_arm, axis=1))
    avg_right_arm_length = np.nanmean(np.linalg.norm(right_arm, axis=1))

    avg_arm_length = (avg_left_arm_length + avg_right_arm_length) / 2

    joints /= avg_arm_length
    joints *= arm_length_in_cm

    return joints

def get_useful_joints_UP2(joints):
    """
    Returns only the useful joints for the UP2 dataset.
    Args:
        joints (np.ndarray): Shape (frames, joints, 2) .. the original joints from mediapipe
            need to be used
    Returns:
        np.ndarray: Shape (frames, useful_joints, 2) .. the useful joints for the UP2 dataset
    """
    useful_points = [
        get_joint_index["nose"],
        get_joint_index["left elbow"],
        get_joint_index["right elbow"],
        get_joint_index["left wrist"],
        get_joint_index["right wrist"],
        get_joint_index["left thumb"],
        get_joint_index["right thumb"],
        # get_joint_index["left shoulder"],
        # get_joint_index["right shoulder"],
    ]
    return joints[:, useful_points, :]

def smooth_joints_UP2(joints, SIGMA):
    """
    Smooths the joints for the UP2 dataset.
    Args:
        joints (np.ndarray): Shape (frames, joints, 2)
        SIGMA (int): The SIGMA value for the gaussian filter
    Returns:
        np.ndarray: Shape (frames, joints, 2) .. the smoothed joints
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
    Detects jumps in the signal.
    Args:
        signal (np.ndarray): The NOT SMOOTHED POSITION (not velocity) signal to detect jumps in. Shape (frames, joints, 2)
        threshold (float): The threshold for the jump detection.
    Returns:
        np.ndarray: The signal with jumps detected. Shape (frames,)
    """
    jumps = np.zeros(signal.shape[0])
    for i in range(1, signal.shape[0]):
        # jump either in x or y coordinate
        for j in range(signal.shape[1]):
            if np.abs(signal[i][j][0] - signal[i - 1][j][0]) > threshold or np.abs(signal[i][j][1] - signal[i - 1][j][1]) > threshold:
                jumps[i] = 1
    return jumps

def process_kinematics_file(joints, visualize=False):

    joints = pixels2cm_using_armlength(joints, ARM_LENGTH_IN_CM)
    joints = get_useful_joints_UP2(joints)
    jumps = detect_jumps(joints, JUMP_THRESHOLD_CM)

    # for indices where are jumps, set Nans in joints
    for i in range(jumps.shape[0]):
        if jumps[i] == 1:
            joints[i] = np.nan # it broadcasts to all joints and both x and y coordinates

    # 1. take joints and correct the z-coordinate by the average z-coordinate of shoulders (they shouldn't move so it's like a reference)
    gradient = joints.copy()

    # 2. smooth the signal
    for joint in range(gradient.shape[1]):
        gradient[:, joint, 0] = smooth(gradient[:, joint, 0], SIGMA)
        gradient[:, joint, 1] = smooth(gradient[:, joint, 1], SIGMA)
        gradient[:, joint, 2] = smooth(gradient[:, joint, 2], SIGMA)

    # 3. compute the gradient for each joint separately
    final_gradient = np.zeros((joints.shape[0], joints.shape[1]))
    for joint in range(gradient.shape[1]):
        gradient[:, joint, 0] = smooth(np.gradient(gradient[:, joint, 0]), SIGMA)
        gradient[:, joint, 1] = smooth(np.gradient(gradient[:, joint, 1]), SIGMA)
        # 4. decrease the movement of z coordinate by a factor (because it's very noisy)
        gradient[:, joint, 2] = smooth(np.gradient(gradient[:, joint, 2] * Z_FACTOR), SIGMA)

        # 5. compute the size (norm) of the gradient
        if USE_Z_COORDINATE:
            final_gradient[:, joint] = np.linalg.norm(gradient[:, joint, :3], axis=1)
        else:
            final_gradient[:, joint] = np.linalg.norm(gradient[:, joint, :2], axis=1)

    if visualize:
        jumps *= 100
        jumps = np.expand_dims(jumps, axis=1)
        non_smoothed_gradient = joints.copy()
        final_non_smoothed_gradient = np.zeros((joints.shape[0], joints.shape[1]))
        for joint in range(non_smoothed_gradient.shape[1]):
            non_smoothed_gradient[:, joint, 0] = np.gradient(non_smoothed_gradient[:, joint, 0])
            non_smoothed_gradient[:, joint, 1] = np.gradient(non_smoothed_gradient[:, joint, 1])
            final_non_smoothed_gradient[:, joint] = np.linalg.norm(non_smoothed_gradient[:, joint, :], axis=1)


        # we visualize the smoothed, not smoothed signals and the big jumps in coordinates
        nose = np.concatenate((final_gradient[:, :7], final_non_smoothed_gradient[:, :7], jumps), axis=1)
        plot(nose, np.arange(15), names = [
            'nose',
            'left elbow',
            'right elbow',
            'left wrist',
            'right wrist',
            'left pinky',
            'right pinky',
            'nose',
            'left elbow',
            'right elbow',
            'left wrist',
            'right wrist',
            'left pinky',
            'right pinky',
            'jumps'
        ])

    new_in_dataset_signal = []
    for datapoint in final_gradient[:, 0]: # Nans are in all joints, so let's use just one joint
        new_in_dataset_signal.append([datapoint, 0])
    
    return final_gradient, new_in_dataset_signal

def process_folder(input_folder, output_folder, visualize=False):
    """
    Processes the folder with kinematics files.
    Args:
        folder (str): The folder with kinematics files.
        visualize (bool, optional): Whether to visualize the processed data. Defaults to False.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in tqdm(os.listdir(input_folder)):
        if file.endswith("_kinematics.npy") and not file.endswith("_processed_kinematics.npy"):
            # if the _processed_kinematics.npy file already exists, skip it
            if os.path.exists(os.path.join(output_folder, file.replace("_kinematics.npy", "_processed_kinematics.npy"))):
                tqdm.write(f"File {file} already processed. Skipping.")
                continue
            joints = np.load(os.path.join(input_folder, file))
            processed_joints, in_dataset = process_kinematics_file(joints, visualize)
            np.save(os.path.join(output_folder, file.replace("_kinematics.npy", "_processed_kinematics.npy")), processed_joints)
            np.save(os.path.join(output_folder, file.replace("_kinematics.npy", "_processed_in_dataset.npy")), in_dataset)

SIGMA = 1 # shortens the signal by 8 frames (by two applications of smoothing)
ARM_LENGTH_IN_CM = 30
JUMP_THRESHOLD_CM = 10 # without prior smoothing! So it's higher because there can be noise
Z_FACTOR = 1/10 # decrease the movement of z coordinate by this factor
USE_Z_COORDINATE = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the kinematics signals.")
    parser.add_argument("extraction_folder", help="Path to the folder containing kinematics .npy files.")
    parser.add_argument("output_folder", help="Path to the output folder to save the processed kinematics signals with processed_in_dataset.npy.")
    args = parser.parse_args()

    process_folder(args.extraction_folder, args.output_folder, visualize=False)

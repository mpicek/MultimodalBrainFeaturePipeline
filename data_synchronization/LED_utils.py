import numpy as np
from skimage import filters, morphology
from scipy import signal
import matplotlib.pyplot as plt
import cv2
import pyrealsense2 as rs

def get_LED_mask(video_array, visualize_pipeline=False, cropped_LED_image_colorful=None, save_vizualization_to=None):
    """
    Generates a mask for LED region in a video based on frame variability.

    Processes a grayscale video array to highlight areas with high variability (likely LED regions because it's blinking)
    through standard deviation calculation. 
    Applies 
     - blurring
     - Otsu's thresholding
     - erosion
     - dilation
    to refine the LED mask. Optionally visualizes the processing pipeline.

    Parameters:
    - video_array (np.ndarray): 3D array of shape (n_frames, height, width), representing the video.
    - visualize_pipeline (bool): If True, visualizes intermediate processing steps.
    - cropped_LED_image_colorful (np.ndarray): A colorful image of the cropped LED (just for visualization purposes).

    Returns:
    - np.ndarray: 2D boolean array (a mask) indicating LED regions.
    """

    std_image = np.std(video_array, axis=0)
    std_image = np.uint8(np.round(std_image))

    # Step 0: Blur the image to reduce noise
    blurred_image = filters.gaussian(std_image, sigma=1)  # Adjust sigma as needed

    # Apply Otsu's method to find the optimal threshold after blurring
    threshold_value = filters.threshold_otsu(blurred_image)

    # Segment the image using the threshold value
    binary_image = blurred_image > threshold_value

    # Post-processing: Apply erosion to refine the binary image
    eroded_image = morphology.erosion(binary_image, morphology.disk(3))  # Adjthen I do some image processing stuff: blurring, segmentation, erosion, dilatation and voilaust the structuring element as needed
    
    dilated_image = morphology.dilation(eroded_image, morphology.disk(3))  # Adjust the structuring element as needed

    # use the dilated image if there is at least one white pixel, otherwise return the binary_image
    final_mask = dilated_image if np.sum(dilated_image) > 0 else binary_image

    if visualize_pipeline:
        visualize_LED_detection_pipeline(std_image, blurred_image, binary_image, final_mask, cropped_LED_image_colorful, save_vizualization_to=save_vizualization_to)

    return final_mask

def visualize_LED_detection_pipeline(std_image, blurred_image, binary_image, dilated_image, cropped_LED_image_colorful=None, save_vizualization_to=None):
    """
    Visualizes the steps in the LED detection pipeline.

    Parameters:
    - std_image (np.ndarray): The original image or the image after applying a standard deviation filter,
                              represented in grayscale.
    - blurred_image (np.ndarray): The image after applying a blurring filter, represented in grayscale.
    - binary_image (np.ndarray): The image after thresholding, represented in binary (black and white).
    - dilated_image (np.ndarray): The image after applying dilation, enhancing features in the binary image.
    - cropped_LED_image_colorful (np.ndarray, optional): The cropped image of the LED signal in BGR format.
                                                          If provided, it is included in the visualization.
    """

    # Visualize the processing steps
    if cropped_LED_image_colorful is None:
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        ax = axes.ravel()

        ax[0].imshow(std_image, cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(blurred_image, cmap='gray')
        ax[1].set_title('Blurred Image')
        ax[1].axis('off')

        ax[2].imshow(binary_image, cmap='gray')
        ax[2].set_title('Binary Image')
        ax[2].axis('off')

        ax[3].imshow(dilated_image, cmap='gray')
        ax[3].set_title('Labeled Image after Erosion')
        ax[3].axis('off')

        plt.tight_layout()
        plt.show()
        if save_vizualization_to:
            plt.savefig(save_vizualization_to)


    else:
        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        ax = axes.ravel()

        ax[0].imshow(cv2.cvtColor(cropped_LED_image_colorful, cv2.COLOR_RGB2BGR))
        ax[0].set_title('Cropped LED')
        ax[0].axis('off')

        ax[1].imshow(std_image, cmap='gray')
        ax[1].set_title('Variance')
        ax[1].axis('off')

        ax[2].imshow(blurred_image, cmap='gray')
        ax[2].set_title('Blurred Image')
        ax[2].axis('off')

        ax[3].imshow(binary_image, cmap='gray')
        ax[3].set_title('Binary Image')
        ax[3].axis('off')

        ax[4].imshow(dilated_image, cmap='gray')
        ax[4].set_title('Binary Image after Erosion and Dilation')
        ax[4].axis('off')

        plt.tight_layout()
        plt.show()
        if save_vizualization_to:
            plt.savefig(save_vizualization_to)


def get_LED_signal_from_video(video_path, LED_mask, ref_point, n_frames, downscale_factor):
    """
    Extracts the LED signal from a video, given an LED mask (for the cropped area)
    and reference points in the original video (to be cropped).

    This function reads a specified number of frames from a video, crops each frame based on provided
    reference points, downscales the cropped frame (does not downsample the FPS), 
    and calculates the average intensity of pixels within the LED mask area.

    Parameters:
    - video_path (str): Path to the video file.
    - LED_mask (np.ndarray): A 2D boolean array representing the mask of the LED area. The mask should
                             have the same dimensions as the downsampled, cropped frame.
    - ref_point (tuple): A tuple of two tuples, each containing (x, y) coordinates, representing the
                         top-left and bottom-right corners of the cropping rectangle.
    - n_frames (int): The number of frames to process from the video.
    - downscale_factor (int): The factor by which the cropped frame is downsampled. For example, a
                              downscale_factor of 2 will reduce both the width and height by half.

    Returns:
    - nd.array: A numpy array of intensity values for the LED area in each processed frame. If the video
            cannot be opened or there are no selected pixels, None is returned.

    Note:
    - This function specifically analyzes the red channel of the video frames.
    - Ensure that the LED_mask dimensions match the dimensions of the downsampled, cropped frame.
    """

    # Ensure the mask is boolean for indexing
    LED_mask = LED_mask.astype(bool)
    
    # Initialize an empty list to store the average values for each frame
    average_values = []

    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    while len(average_values) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames to read
        
        # crop the image
        cropped = frame[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
        
        # Append the resized frame to the list
        frame = cropped[..., 2][::downscale_factor, ::downscale_factor] # only red channel of the array
        selected_pixels = frame[LED_mask]

        # Compute the average of the selected pixels
        # Check if selected_pixels is not empty to avoid division by zero
        if selected_pixels.size > 0:
            average_value = np.mean(selected_pixels)
        else:
            average_value = 0  # Or np.nan, depending on how you want to handle frames with no selected pixels
        
        # Append the average value to the list
        average_values.append(average_value)

    cap.release()
    
    return np.array(average_values)

def get_LED_signal_from_array(video_array, LED_mask):
    """
    Calculates the average LED signal intensity over time from a video array.

    Parameters:
    - video_array (np.ndarray): Video data as (n_frames, height, width).
    - LED_mask (np.ndarray): Boolean mask for the LED area (height, width).

    Returns:
    - np.ndarray: Average intensity of the LED area per frame (n_frames,).
    """

    # Ensure the mask is boolean for indexing
    LED_mask = LED_mask.astype(bool)
    video_array = video_array[:, LED_mask]
    average_values = np.mean(video_array, axis=1)
    
    return average_values

def compute_offset(realsense_LED_signal, camera_LED_signal):
    """
    Computes the time offset between two LED signals using cross-correlation.

    This function calculates the cross-correlation between an LED signal from a Realsense device
    and an LED signal from a camera to determine the time offset between them. The time offset
    indicates how much the camera LED is delayed behind realsense LED.

    Parameters:
    - realsense_LED_signal (array_like): The LED signal captured by the Realsense device.
    - camera_LED_signal (array_like): The LED signal captured by the camera.

    Returns:
    - lag (int): How much the camera's LED is delayed behind realsense LED.
                 The starts are at realsense_LED_signal[0] and camera_LED_signal[-lag]
                 or (equivalently) realsense_LED_signal[-lag] = camera_LED_signal[0] (but here
                 it goes into the past if lag > 0 which is nonexistent, so don't use this because it would
                 roll over to the end of realsense_LED_signal, which is not correct)
    """

    correlation = np.correlate(realsense_LED_signal, camera_LED_signal, mode='full')
    lags = signal.correlation_lags(realsense_LED_signal.size, camera_LED_signal.size, mode="full")
    lag = lags[np.argmax(correlation)]

    return lag


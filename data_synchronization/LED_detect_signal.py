import numpy as np
from skimage import io, filters, measure, color, morphology
from scipy import signal
import matplotlib.pyplot as plt
from LED_video_crop import crop_subsampled_LED_red_channel_from_video_for_std
import cv2

def get_LED_mask(video_array, visualize_pipeline=False):
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

    if visualize_pipeline:

        # Visualize the processing steps
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
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

    return dilated_image

def get_LED_signal(video_path, LED_mask, ref_point, n_frames, downscale_factor):
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


if __name__ == "__main__":
    video_path = '/home/mpicek/repos/master_project/test_data/camera/C0359.MP4'
    n_frames = 10000
    downscale_factor = 2
    downsample_frames_factor = 25
    subsampled_video_array, ref_point = crop_subsampled_LED_red_channel_from_video_for_std(video_path, n_frames, downscale_factor, downsample_frames_factor)
    binary_mask = get_LED_mask(subsampled_video_array, visualize_pipeline=True)

    if subsampled_video_array is not None:
        print(f"Resized video array shape: {subsampled_video_array.shape}")
        # The shape will be (num_frames, new_height, new_width, channels)
    
    average_values = get_LED_signal(video_path, binary_mask, ref_point, n_frames, downscale_factor)
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(average_values)) / 25, average_values)
    plt.xlabel('Time (s)')
    plt.ylabel('Average pixel value')
    plt.title('Average pixel value over time')
    plt.show()
import numpy as np
from LED_utils import get_LED_mask, get_LED_signal_from_video
import cv2
from LED_GUI_cropper import ImageCropper
import os

def crop_subsampled_LED_red_channel_from_video_for_std(video_path, n_frames, downscale_factor, downsample_frames_factor):
    """
    Crops the red channel of the LED area from the first n_frames of a video.
    The resulting video is VERY subsampled and is supposed to be used for the std calculation only.

    This function opens a video file, allows the user to select a region of interest (ROI)
    containing an LED in the first frame, and then crops this ROI from the red channel
    of each of the first n_frames of the video. The cropped areas from each frame are
    stacked into a single NumPy array.

    Parameters:
    - video_path (str): The file path to the video.
    - n_frames (int): The number of frames from the start of the video to process.
    - downscale_factor (int): By how much to downsample the cropped image of the LED (to save space)

    Returns:
    - tuple: subsampled_video_array (np.ndarray): A 3D NumPy array containing the cropped red channel of the selected ROI
                  from every ith frame of the first n_frames (where i = downsample_frames_factor).
                  The array shape is (n_frames // downsample_frames_factor, height // downscale_factor, width // downscale_factor),
                  where height and width correspond to the dimensions of the cropped area.
                  Returns None if the video cannot be opened or if no ROI is selected.
             ref_point: A list of reference points defining the selected rectangle (in the original video).
             cropped_LED_image_colorful (np.ndarray): the cropped LED from the first frame 
                  (all channels, thus shape (height, width, 3))

    Note:
    - The function uses an ImageCropper object to allow the user to select the ROI on the first frame.
      The same ROI is then used for cropping the subsequent frames.
    - Only the red channel of the cropped area is extracted and returned.
    - If the specified number of frames (n_frames) is greater than the total number of frames in the
      video, the function will process only the available frames.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    cropped_LED_image_colorful = None
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    frames = []  # List to hold the cut frames

    frame_counter = 0

    num_of_frames_of_the_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while frame_counter < n_frames:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames to read
        
        frame_counter += 1
        if frame_counter < 30 and n_frames >= 30 and num_of_frames_of_the_video >= 30:
            continue

        if len(frames) == 0:
            frame_counter = 0
            # Create an ImageCropper object
            cropper = ImageCropper(frame)
            cropped_LED_image_colorful, ref_point = cropper.show_and_crop_image()

        if frame_counter % downsample_frames_factor == 0:
            # crop the image
            cropped = frame[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            
            # Append the resized frame to the list
            frames.append(cropped[..., 2][::downscale_factor, ::downscale_factor]) # only red channel of the array
    

    # Stack the frames into a single array
    subsampled_video_array = np.stack(frames, axis=0)
    # Release the video capture object
    cap.release()
    
    return subsampled_video_array, ref_point, cropped_LED_image_colorful


if __name__ == "__main__":
    """
    INPUT:  /path/to/video/video.mp4
    OUTUPT: /path/to/video/video_LED_data.npy
            /path/to/video/video_LED_mask.png  (if the visualization is enabled in get_LED_mask function)
    """

    # video_path = '/data/output.mp4'
    video_path = '/data/bags/cam0_916512060805_record_04_10_2023_1341_07.mp4'
    video_basename = os.path.basename(video_path)[:-4]
    video_path_only = os.path.dirname(video_path)

    n_frames = 54000 # (54000 = 30 minutes of 30 fps video)
    downscale_factor = 2
    downsample_frames_factor = 25

    subsampled_video_array, ref_point, cropped_LED_image_colorful = crop_subsampled_LED_red_channel_from_video_for_std(
        video_path,
        n_frames,
        downscale_factor,
        downsample_frames_factor
    )

    binary_mask = get_LED_mask(
        subsampled_video_array,
        visualize_pipeline=True,
        cropped_LED_image_colorful=cropped_LED_image_colorful,
        save_vizualization_to=os.path.join(video_path_only, video_basename + '_LED_mask.png')
    )

    if subsampled_video_array is not None:
        print(f"Resized video array shape: {subsampled_video_array.shape}")
        # The shape will be (num_frames, new_height, new_width, channels)
    
    average_values = get_LED_signal_from_video(video_path, binary_mask, ref_point, n_frames, downscale_factor)

    np.save(os.path.join(video_path_only, video_basename + '_LED_data.npy'), average_values)

    
    # np.save('/data/output.npy', average_values)
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(average_values)) / 25, average_values)
    plt.xlabel('Time (s)')
    plt.ylabel('Average pixel value')
    plt.title('Average pixel value over time')
    plt.show()
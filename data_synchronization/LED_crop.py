import cv2
import numpy as np
from LED_GUI import ImageCropper

def crop_LED_red_channel_from_video(video_path, n_frames):
    """
    Crops the red channel of the LED area from the first n_frames of a video.

    This function opens a video file, allows the user to select a region of interest (ROI)
    containing an LED in the first frame, and then crops this ROI from the red channel
    of each of the first n_frames of the video. The cropped areas from each frame are
    stacked into a single NumPy array.

    Parameters:
    - video_path (str): The file path to the video.
    - n_frames (int): The number of frames from the start of the video to process.

    Returns:
    - np.ndarray: A 3D NumPy array containing the cropped red channel of the selected ROI
                  from each of the first n_frames. The array shape is (n_frames, height, width),
                  where height and width correspond to the dimensions of the cropped area.
                  Returns None if the video cannot be opened or if no ROI is selected.

    Note:
    - The function uses an ImageCropper object to allow the user to select the ROI on the first frame.
      The same ROI is then used for cropping the subsequent frames.
    - Only the red channel of the cropped area is extracted and returned.
    - If the specified number of frames (n_frames) is greater than the total number of frames in the
      video, the function will process only the available frames.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    frames = []  # List to hold the cut frames
    
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames to read
        
        if len(frames) == 0:
            # Create an ImageCropper object
            cropper = ImageCropper(frame)
            _, ref_point = cropper.show_and_crop_image()

        # crop the image
        cropped = frame[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
        
        # Append the resized frame to the list
        frames.append(cropped[..., 2]) # only red channel of the array
    
    # Stack the frames into a single array
    video_array = np.stack(frames, axis=0)
    # Release the video capture object
    cap.release()
    
    return video_array

if __name__ == "__main__":
    video_path = '/home/mpicek/repos/master_project/test_data/camera/C0172.MP4'
    # Example usage
    video_array = crop_LED_red_channel_from_video(video_path, 100)

    if video_array is not None:
        print(f"Resized video array shape: {video_array.shape}")
        # The shape will be (num_frames, new_height, new_width, channels)
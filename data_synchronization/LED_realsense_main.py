import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyrealsense2 as rs
from LED_GUI_cropper import ImageCropper
from LED_utils import get_LED_mask, get_LED_signal_from_array

def get_LED_signal_realsense(path_bag, downscale_factor):
    """
    Extracts and downscales the LED signal from a RealSense camera .bag file.

    This function processes a video file captured by a RealSense camera, identifies the LED signal
    in the first frame, and then extracts and downscales this signal from the entire video.

    Parameters:
    - path_bag (str): Path to the .bag file captured by the RealSense camera.
    - downscale_factor (int): Factor by which the LED signal image is downscaled. A factor of 2 
                              would reduce both the height and width of the image by half.

    Returns:
    - LED_video (np.ndarray): A stack of downsampled grayscale images of the LED signal throughout 
                              the video. The shape is (n_frames, height//downscale_factor, 
                              width//downscale_factor), where n_frames is the number of frames in 
                              the video.
    - cropped_LED_image_colorful (np.ndarray): The cropped image of the LED signal from the first 
                                               frame in BGR format with shape (height, width, 3).

    Note:
    The function assumes the video's resolution is 640x480 and processes frames assuming this 
    resolution. It sets RealSense playback to real-time mode initially to avoid errors at the start 
    and end of the video but processes frames as slowly as necessary by disabling real-time playback 
    after the first frame.
    """

    img_height, img_width = 480, 640

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, path_bag)

    config.enable_stream(rs.stream.depth, img_width, img_height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, img_width, img_height, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    
    # we want to set_real_time to False, so that we can process the frames as slowly as necessary.
    # However, there are errors reading it at the beginning and at the end (hopefully not in the middle)
    # when we set it to False right at the beginning. Therefore, we set it to True at the beginning 
    # and then to False after the first frame is read. The end is handled differently - with reading the frames
    # up to the duration of the video - 1s (to be sure that we don't miss the last frame)
    playback.set_real_time(True)

    first_frame = True
    prev_ts, frame_nb = -1, -1
    max_frame_nb = 0

    LED_video, time_series = [], []
    ref_point, cropped_LED_image_colorful = None, None

    duration = playback.get_duration().total_seconds() * 1000
    print(f"Overall video duration: {playback.get_duration()}")

    try:
        while True:
            print("beginning ", frame_nb, max_frame_nb)
            try:
                frames = pipeline.wait_for_frames()
            except:
                print("READING ERROR")
                playback.set_real_time(True)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            playback.set_real_time(False)

            # skipped frame by RealSense
            if not depth_frame or not color_frame:
                print("SKIPPED FRAME! HANDLE IT!")
                continue

            frame_nb = color_frame.get_frame_number()
            print("after getting frame_nb ", frame_nb, max_frame_nb)

            # end of the file (no more new frames)
            if frame_nb < max_frame_nb:
                print(frame_nb, max_frame_nb)
                break

            max_frame_nb = frame_nb

            ts = frames.get_timestamp()

            depth_image_rs = np.asanyarray(depth_frame.get_data())
            color_image_rs = np.asanyarray(color_frame.get_data()) # RGB!!!

            if ref_point is None: # aka it's the first frame being processed
                cropper = ImageCropper(color_image_rs)
                cropped_LED_image_colorful, ref_point = cropper.show_and_crop_image()
                print(ref_point)

            cropped = color_image_rs[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            cropped_downscaled_red = cropped[..., 2][::downscale_factor, ::downscale_factor]
            LED_video.append(cropped_downscaled_red.copy())

            if first_frame: 
                t0 = ts
                first_frame = False
            
            # the video is at the end (without the last second) so we kill the reading
            if ts - t0 + 1000 > duration:
                break

            # if we have already processed this frame (duplicate, it's an error of realsense that has to be handled)
            if prev_ts >= int(ts-t0): 
                continue

            # Wait for a key press and close all windows
            # cv2.waitKey(1)

            prev_prev_ts = prev_ts
            prev_ts = int(ts-t0)

            if prev_ts - prev_prev_ts > 2*int(1000/30) - 10:
                print(f"Skipped frame. Previous ts: {prev_ts}, difference {prev_ts - prev_prev_ts}")

            time_series.append([prev_ts])

            ch = cv2.waitKey(1)
            if ch==113: #q pressed
                break

            if len(time_series) % (60*30) == 0:
                print(f"{len(time_series) / (60*30)} minutes processed")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        LED_video = np.stack(LED_video)
    
    return LED_video, cropped_LED_image_colorful


if __name__ == "__main__":

    path_bag = "/home/mpicek/repos/MultimodalBrainFeaturePipeline/test_data/corresponding/cam0_911222060374_record_13_11_2023_1337_20.bag"
    output_file = "LED_20.npy"
    # path_bag = "/home/mpicek/repos/MultimodalBrainFeaturePipeline/test_data/corresponding/cam0_911222060374_record_13_11_2023_1330_19.bag"

    downscale_factor = 1 # realsense is 640x480, so we don't need to downscale it
    video_array, cropped_LED_image_colorful = get_LED_signal_realsense(path_bag, downscale_factor)
    binary_mask = get_LED_mask(video_array, visualize_pipeline=True, cropped_LED_image_colorful=cropped_LED_image_colorful)

    if video_array is not None:
        print(f"Resized video array shape: {video_array.shape}")
        # The shape will be (num_frames, new_height, new_width, channels)
    
    average_values = get_LED_signal_from_array(video_array, binary_mask)
    # np.save(output_file, average_values)
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(average_values)) / 25, average_values)
    plt.xlabel('Time (s)')
    plt.ylabel('Average pixel value')
    plt.title('Average pixel value over time')
    plt.show()

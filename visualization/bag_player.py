import pyrealsense2 as rs
import numpy as np
import cv2
import time

load_dir = '/home/mpicek/repos/master_project/test_data/corresponding/'
filename = 'cam0_911222060374_record_13_11_2023_1337_20.bag'
# filename = 'cam0_911222060374_record_13_11_2023_1330_19.bag'
save_dir = '/media/mpicek/T7/martin/bags/lol/'

pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, '/media/mpicek/T7/martin/bags/cam0_911222060374_record_06_09_2023_1440_06.bag')
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# pipeline.start(config)

first = True
prev_ts = -1
max_frame_nb = 0

# change this to True if you want to run the recording faster than real time
real_time = True

if real_time:
    pipeline.start(config)
else:
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

try:
    while True:

        
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        frame_nb = color_frame.get_frame_number()

        if frame_nb < max_frame_nb:
            break #FIXME

        max_frame_nb = frame_nb

        ts = frames.get_timestamp()

        if first: 
            t0 = ts
            first = False

        # Convert images to numpy arrays
        depth_image_1 = np.asanyarray(depth_frame.get_data())
        color_image_1 = np.asanyarray(color_frame.get_data())

        if prev_ts >= int(ts-t0):
            continue

        prev_ts = int(ts-t0)

        # Show images from both cameras
        cv2.imshow('RealSense', color_image_1)

        # Save images and depth maps from both cameras by pressing 's'
        ch = cv2.waitKey(1)
            
        if ch==113: #q pressed
            break

        # delay 1s
        # time.sleep(1)

finally:

    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
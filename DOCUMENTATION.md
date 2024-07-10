# Data synchronization

### Prerequisite: A Trained DLC Network

#### 0a. Extract the dataset for DLC training (DLC/extract_frames_for_dataset.py)

#### 0b. Train the DLC network (DLC/train_network.ipynb)


# Pipeline


#### 1. Download .bag files and the corresponding wisci (to bags/ and wisci/)
 - wisci can be downloaded from the server using `copy_only_mat_files.py. And careful, comment out or adjust the condition on line 24!

#### 2. bag2mp4.py

Batch conversion. Appends to the log table if exists. Skips already converted videos (found by basename, not full path).

## Movement synchronization

#### 1. Extract movement from the videos (DLC/analyze_video.py)

Read the README.md in `synchronization_docker/` to see how to run the docker container for this step. 
Into `mp4` folder. Skips already analyzed videos in that folder

#### 2. Synchronize the movement from videos with WISCI files (Synchronizer.py or ParallelSynchronizer.py for parallel processing)

## LED Synchronization

#### 1. Extract LED positions from the videos (extract_LED_position.py)
- Select area and press 'c' to continue (or 'r' to reset the area).

#### 2. Extract LED signal from the .mp4 videos (get_LED_signal.py)

#### 3. Synchronize the LED signal with WISCI files (LEDSynchronizer.py)

#### 4. Quality Control: Manually filter properly synchronized data with LED (LED_sync_quality_test.py)

Notes into log file into a column passed_quality_test whether it passed the quality test.
It will create the column if doesn't exist yet.
 - to mark that the sync passed the test, press right arrow
 - to mark that the sync did not pass the test, press left arrow

## Comparison of LED and Acc sync
 - with `led_acc_comparison.ipynb`

## Dataset extraction

#### Detection of occlusions & finding the patient's bounding box
 - crop_patient_and_detect_occlusions.py
 - Finds patient in the video based on the detection with YOLO and the nose position from DLC
 - returns `_bounding_box.npy`, `_occlusion.npy`, `_cropped.png` (for manual validation), and `skipped_files.csv` (logs only files that were skipped). We skip the files for the following reasons:
      - the nose position likelihood is too low (so the patient is not there or he/she is backwards or DLC just didn't detect it) - when likelihood is < 0.4
      - no person is at the location of the detected patient's nose
      - no people detected in the whole video

    - bounding box is in format (x1, y1, x2, y2), it's in pixels and x increases from left to right on the image, y increases from top to bottom ([0, 0] is top left in the image)
    - occlusion can be: 
       - 0: no occlusion
       - 1: full occlusion or multiple small occlusions
       - 2: occlusion from the right on image (left side of patient)
       - 3: occlusion from the left

#### Crop patient from the videos 
 - crop_videos.py
 - takes the bounding boxes from the previous script and crops the videos according to them
 - outputs `_cropped.mp4`

#### Extract the kinematics
 - extract_kinematics.py
 - the input are the cropped videos
 - returns `_kinematics.npy`, `_annotated.mp4` (for manual (visual) validation) and `_in_dataset.npy`
 - `_in_dataset.npy` has for each frame value:
    - 0 if there is occlusion of the whole body with the patient OR mediapipe didn't detect the patient
    - 1 if there is no occlusion or just partial one AND mediapipe successfully detected the patient
 - Sets Nan values to occluded joints, otherwise it sets the coordinate **in pixel space**
   - So it is possible that some of the joints in the same frame are Nan and some are not
 - returns all the detected joints (33 of them) in format [x, y, z, visibility]
 - in the script, set parameters of mediapipe. The defaults we use are: num_poses (5), min_pose_detection_confidence (0.8), min_tracking_confidence (0.7)

#### Preprocess kinematics
 - preprocess_2d_kinematics.py
 - returns `_processed_2d_kinematics.npy` and `_processed_2d_in_dataset.npy`
 - the input is the kinematics from the previous step
 - filters out the unimportant joints, returns only 9 of them 
 - detects jumps for each joint separately (caused by physios' hands and low-quality mediapipe detection) - using normalization to shoulder length and filters out these jumps (sets the coordinates to Nan)
 - Filters out movements when the joints have visibility lower than 0.75 (from mediapipe)
 - Smooths the signals (only 2D positions) but does not shorten it
 - DOES NOT COMPUTE GRADIENT OF JOINTS' MOVEMENTS (the older code for that is commented)

 #### Segment the patient
  - segment_pose.py or segment_pose_PARALLEL.py
  - the input are the cropped videos
  - returns `_segmented_pose.mp4` and `_segmented_pose_temp_in_dataset.npy`
  - segments the patient and puts a green background behind him 
  - uses Mediapipe pose estimation because it can also return the segmented body

#### Extract the DINO features
 - extract_dino_features.py
 - returns `_features.npy`
 - extracts features from the cropped videos with segmented patient (with green background) using DINO
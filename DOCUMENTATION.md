# Project Documentation

This project is documented via this file and the README.md files in each subfolder. Each script is a standalone script with command line arguments that explain the usage of the script. The internal structure of the scripts are also documented with comments, so to see how the script works, consult the source code. This and the README.md documents are here to provide a high-level overview of the project.

### 1. Get the Data and Prepare the Environment

**Prerequisite: A Trained DLC Network** - An in-depth documentation provided in [README.md](synchronization_docker/README.md) in the folder `synchronization_docker/`.

#### 1a. Extract the dataset for DLC training (DLC/extract_frames_for_dataset.py)

#### 1b. Train the DLC network (DLC/train_network.ipynb)

#### 1c. Download .bag files and the corresponding wisci files
 - WISCI files have to be downloaded from the server. For that, you can use data_synchronization/copy_only_mat_files.py. And be careful, comment out or adjust the condition on line 24!

#### 1d. Convert .bag files to .mp4 files
- data_synchronization/bag2mp4.py
- Batch conversion. Appends to the log table if exists. Skips already converted videos (found by basename, not full path).

### 2. Option 1: Movement Synchronization using Accelerometer Synchronization

#### 2a. Extract movement from the videos (DLC/analyze_video.py)

Read the [README.md](synchronization_docker/README.md) in `synchronization_docker/` to see how to run the docker container for this step. 
Into `mp4` folder. Skips already analyzed videos in that folder

#### 2b. Synchronize the movement from videos with WISCI files (Synchronizer.py or ParallelSynchronizer.py for parallel processing)

### 2. Option 2: Movement Synchronization using LED Synchronization

#### 2a. Extract LED positions from the videos (extract_LED_position.py)
- Select area and press 'c' to continue (or 'r' to reset the area).

#### 2b. Extract LED signal from the .mp4 videos (get_LED_signal.py)

#### 2c. Synchronize the LED signal with WISCI files (LEDSynchronizer.py)

#### 2d. Quality Control: Manually filter properly synchronized data with LED (LED_sync_quality_test.py)

Logs into log file into a column passed_quality_test whether it passed the quality test.
It will create the column if doesn't exist yet.
 - to mark that the sync passed the test, press right arrow
 - to mark that the sync did not pass the test, press left arrow

### 3. (Optional) Comparison of LED and Acc sync
 - with `led_acc_comparison.ipynb`

### 4. Dataset extraction

#### 4a. Detection of occlusions & finding the patient's bounding box
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

#### 4b. Crop patient from the videos 
 - crop_videos.py
 - takes the bounding boxes from the previous script and crops the videos according to them
 - outputs `_cropped.mp4`

#### 4c. Extract the kinematics
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

#### 4d. Preprocess kinematics
 - preprocess_2D_kinematics.py
 - returns `_processed_2d_kinematics.npy` and `_processed_2d_in_dataset.npy`
 - the input is the kinematics from the previous step
 - filters out the unimportant joints, returns only 9 of them 
 - detects jumps for each joint separately (caused by physios' hands and low-quality mediapipe detection) - using normalization to shoulder length and filters out these jumps (sets the coordinates to Nan)
 - Filters out movements when the joints have visibility lower than 0.75 (from mediapipe)
 - Smooths the signals (only 2D positions) but does not shorten it
 - DOES NOT COMPUTE GRADIENT OF JOINTS' MOVEMENTS (the older code for that is commented)

 #### 4e. Segment the patient
  - segment_pose.py or segment_pose_PARALLEL.py
  - the input are the cropped videos
  - returns `_segmented_pose.mp4` and `_segmented_pose_temp_in_dataset.npy`
  - segments the patient and puts a green background behind him 
  - uses Mediapipe pose estimation because it can also return the segmented body

#### 4f. Extract the DINO features
 - extract_dino_features.py
 - returns `_features.npy`
 - extracts features from the cropped videos with segmented patient (with green background) using DINO

#### 4g. Extract the K-DINO features
  - extract_K-DINO_features.py
  - needs a pretrained network trained in experiments/predict_kinematics_from_dino.ipynb (described bellow)
  - returns `_K-DINO_features.npy`

### 5. Dataset Generation
 - connect_modalities_into_dataset.py
 - IMPORTANT NOTICE: A hardcoded array of the days for testing (test_dates). Change it accordingly. Firstly run the script for the training dataset generation, then inverse the condition in the code to generate the test dataset.
 - For videos that passed synchronization (change the code according to what synchronization you are interested in), it connects all modalities into a dataset. For each video, there is a sliding window of 1s length that jumps by 15 frames (~500ms). It connects the wavelet transformed brain features, DINO features, kinematics and accelerometer within this one second into one `.pkl` file. At also puts there metadata such as what index was used in each of the modality etc.
 - It uses relative indexing, so all the indices are normalized accoring to th wisci signal length.

### 7. Experiments
 - documented in the file [README.md](experiments/README.md) in the folder `experiments/`

### 6. Testing the Decoder on features from different models
 - documented in the file [README.md](decoder_testing/README.md) in the folder `decoder_testing/`


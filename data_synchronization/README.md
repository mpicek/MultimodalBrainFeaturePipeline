# Data synchronization

### Prerequisite: A Trained DLC Network

#### 0a. Extract the dataset for DLC training (DLC/extract_frames_for_dataset.py)

#### 0b. Train the DLC network (DLC/train_network.ipynb)


# Pipeline


#### 1. Download .bag files and the corresponding wisci (to bags/ and wisci/)

#### 2. bag2mp4.py

Batch conversion. Appends to the log table if exists. Skips already converted videos (found by basename, not full path).

## Movement synchronization

#### 1. Extract movement from the videos (DLC/analyze_video.py)

Into `mp4` folder. Skips already analyzed videos in that folder

#### 2. Synchronize the movement from videos with WISCI files (Synchronizer.py or ParallelSynchronizer.py for parallel processing)

## LED Synchronization

#### 1. Extract LED positions from the videos (extract_LED_position.py)

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
 - returns `_bounding_box.npy`, `_occlusion.npy`, `_cropped.png` (for manual validation), and `skipped_files.csv` (just a log to not process the skipped files more times)

#### Crop patient from videos 
 - crop_videos.py
 - for computing higher semantic features with DinoV2/
 - outputs `_cropped.mp4`

#### Extract the kinematics
 - extract_kinematics.py
 - returns `_kinematics.npy`, `_annotated.mp4` (for manual validation) and `_in_dataset.npy`
 - `_in_dataset.npy` has for each frame value:
    - 0 if there is occlusion with the patient OR mediapipe didn't detect the patient
    - 1 if there is no occlusion AND mediapipe successfully detected the patient

#### Preprocess kinematics
 - preprocess_kinematics.py
 - returns `_processed_kinematics.npy` and `_processed_in_dataset.npy`
 - selects the important joints, detects jumps (caused by physios' hands and low-quality mediapipe detection), smooths the signals
 - then computes gradient of each joint
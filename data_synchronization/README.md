# Data synchronization

### Prerequisite: A Trained DLC Network


#### 0a. Extract the dataset for DLC training (DLC/extract_frames_for_dataset.py)

#### 0b. Train the DLC network (DLC/dlc_train_network.ipynb)


## Pipeline

#### 1. Download .bag files and the corresponding wisci (to bags/ and wisci/)

#### 2. bag2mp4.py

Batch conversion. Appends to the log table if exists. Skips already converted videos (found by basename, not full path).

#### 3. Extract movement from the videos (DLC/dlc_analyze_video.py)

Into `mp4` folder. Skips already analyzed videos in that folder

#### OPTIONAL: Extract LED signal from the .mp4 videos (extract_LED_position.py, then get_LED_signal.py)

#### 4. Synchronize the movement from videos with WISCI files (Synchronizer.py)

#### OPTIONAL: Synchronize the LED signal with WISCI files (LedSynchronizer.py)

#### OPTIONAL: Compare the LED sync with the movement sync (led_acc_comparison.ipynb)
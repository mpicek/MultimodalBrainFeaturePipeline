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

# Experiments

For the experiments with RCDM, you need different dependencies that can be found in the RCDM repository (look in the directory).

**Do not forget to set the `MODEL_FLAGS_128` environment variable accoring to the README.md in the RCDM subrepository**
`export MODEL_FLAGS_128="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"`

On the computational resources used for this project, the RCDM conda environment is set up.

Some random command I found in history (this is approximately how to use RCDM):

python RCDM_sample_from_video.py $MODEL_FLAGS_128 --features_path /media/cyberspace007/T7/martin/data/dino_features_0_7/cam0_916512060805_record_21_09_2023_1459_37_features.npy --index 1110 --subtract_from_frame 1080 --batch_size 3


## Documentation

### MLP: DINO -> Kinematics

- DINO features and kinematics are always centerd within the day (the day's mean is subtracted).
- train, val and test data are split by days (whole days are assigned into one of those sets)
- if ANY joint has a NaN value, the whole frame is removed from the dataset
- it is not taking into account the temporal information of the frames (i.e. the order of the frames is not considered). It just predicts the kinematics from the current frame
- It prints how many frames is in each subdataset (train/val/test)
- computes the MSE bound - it is the MSE of the mean of the train set, meaning that if the model predicts the mean of the train set, this is the error that we would get
- INTERESTING RESULT: From the experiments we found out that it is better to center DINO by subtracting the session's mean. Therefore, it can get the kinematics only from the differences of the DINO features.



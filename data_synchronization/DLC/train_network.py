import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

print("------ CHECK THAT GPUS ARE WORKING ^^^ ------")

import deeplabcut

config_path = '/mpicek/dlc_nose_detector/UP2_movement_sync-mpicek-2024-04-15/config.yaml'

deeplabcut.create_training_dataset(config_path, augmenter_type='imgaug')

deeplabcut.train_network(
    config_path, shuffle=1, 
    trainingsetindex=0, 
    gputouse=0, 
    max_snapshots_to_keep=10, 
    autotune=False, 
    displayiters=1000, 
    saveiters=25000, 
    maxiters=500000, 
    allow_growth=True
)


import os
import os.path
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import predict

def prepare_network(
    config,
    directory,
    frametype=".png",
    shuffle=1,
    trainingsetindex=0,
    gputouse=None,
    save_as_csv=False,
    modelprefix="",
):
    """
    THIS FUNCTION IS COPIED FROM DEEPLABCUT PACKAGE. IT'S ORIGINAL NAME IS analyze_time_lapse_frames
    AND I HAVE CUT BIG CHUNK OF IT AS I ONLY NEED TO INITIALIZE THE NETWORK AND RETURN ALL NECESSARY
    VARIABLES FOR FURTHER PREDICTION OF INDIVIDUAL NUMPY FRAMES.

    Analyzed all images (of type = frametype) in a folder and stores the output in one file.

    You can crop the frames (before analysis), by changing 'cropping'=True and setting 'x1','x2','y1','y2' in the config file.

    Output: The labels are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
            in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
            in the same directory, where the video is stored. However, if the flag save_as_csv is set to True, the data can also be exported in \n
            comma-separated values format (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    directory: string
        Full path to directory containing the frames that shall be analyzed

    frametype: string, optional
        Checks for the file extension of the frames. Only images with this extension are analyzed. The default is ``.png``

    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``; if provided it must be either ``True`` or ``False``

    Examples
    --------
    If you want to analyze all frames in /analysis/project/timelapseexperiment1
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml','/analysis/project/timelapseexperiment1')
    --------

    Note: for test purposes one can extract all frames from a video with ffmeg, e.g. ffmpeg -i testvideo.avi thumb%04d.png
    """
    if "TF_CUDNN_USE_AUTOTUNE" in os.environ:
        del os.environ["TF_CUDNN_USE_AUTOTUNE"]  # was potentially set during training

    # if gputouse is not None:  # gpu selection
    #     auxfun_models.set_visible_devices(gputouse)

    tf.compat.v1.reset_default_graph()
    start_path = os.getcwd()  # record cwd to return to this directory in the end

    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg["TrainingFraction"][trainingsetindex]
    modelfolder = os.path.join(
        cfg["project_path"],
        str(
            auxiliaryfunctions.get_model_folder(
                trainFraction, shuffle, cfg, modelprefix=modelprefix
            )
        ),
    )
    path_test_config = Path(modelfolder) / "test" / "pose_cfg.yaml"
    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for shuffle %s and trainFraction %s does not exist."
            % (shuffle, trainFraction)
        )
    # Check which snapshots are available and sort them by # iterations
    try:
        Snapshots = np.array(
            [
                fn.split(".")[0]
                for fn in os.listdir(os.path.join(modelfolder, "train"))
                if "index" in fn
            ]
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."
            % (shuffle, shuffle)
        )

    if cfg["snapshotindex"] == "all":
        print(
            "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!"
        )
        snapshotindex = -1
    else:
        snapshotindex = cfg["snapshotindex"]

    increasing_indices = np.argsort([int(m.split("-")[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

    ##################################################
    # Load and setup CNN part detector
    ##################################################

    # Check if data already was generated:
    dlc_cfg["init_weights"] = os.path.join(
        modelfolder, "train", Snapshots[snapshotindex]
    )
    trainingsiterations = (dlc_cfg["init_weights"].split(os.sep)[-1]).split("-")[-1]

    # update batchsize (based on parameters in config.yaml)
    dlc_cfg["batch_size"] = cfg["batch_size"]

    # Name for scorer:
    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.get_scorer_name(
        cfg,
        shuffle,
        trainFraction,
        trainingsiterations=trainingsiterations,
        modelprefix=modelprefix,
    )
    sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)

    # update number of outputs and adjust pandas indices
    dlc_cfg["num_outputs"] = cfg.get("num_outputs", 1)

    xyz_labs_orig = ["x", "y", "likelihood"]
    suffix = [str(s + 1) for s in range(dlc_cfg["num_outputs"])]
    suffix[0] = ""  # first one has empty suffix for backwards compatibility
    xyz_labs = [x + s for s in suffix for x in xyz_labs_orig]

    pdindex = pd.MultiIndex.from_product(
        [[DLCscorer], dlc_cfg["all_joints_names"], xyz_labs],
        names=["scorer", "bodyparts", "coords"],
    )

    # if gputouse is not None:  # gpu selectinon
    #     auxfun_models.set_visible_devices(gputouse)
    
    return cfg, dlc_cfg, sess, inputs, outputs, pdindex, DLCscorer, DLCscorerlegacy, start_path

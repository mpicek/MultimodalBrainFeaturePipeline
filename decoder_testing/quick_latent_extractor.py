import os
import numpy as np
import h5py
import torch
from tqdm import tqdm
import torch.nn as nn
import scipy.io
from scipy.io import savemat
import argparse
import sys
sys.path.append('/home/cyberspace007/mpicek/NeuralMAE')
import neuralmae.neural_models.models_multimodal_neuralmae_up2 as models_mae_multimodal
import neuralmae.neural_models.models_neuralmae_bsi as models_mae


def prepare_pretrained_brainGPT(CHECKPOINT_PATH, arch):
    #build model
    model = getattr(models_mae, arch)()

    #load model
    chkpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
    msg = model.load_state_dict(chkpt['model'], strict=False)
    print(msg)
    return model

def prepare_pretrained_brainGPT_multimodal(CHECKPOINT_PATH, arch):

    freeze_brainGPT = False
    model = models_mae_multimodal.__dict__[arch](norm_pix_loss=False,
                                                    norm_session_loss=True,
                                                    uniformity_loss=False,
                                                    lamb=0.01,
                                                    # input_size=tuple(args.input_size), 
                                                    # patch_size=tuple(args.patch_size),
                                                    use_projector=False,
                                                    projector_dim=64,
                                                    freeze_brainGPT=freeze_brainGPT)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    return model

def rankme(Z):
    """
    RankMe smooth rank estimation
    from: https://arxiv.org/abs/2210.02885

    Z: (N, K), N: nb samples, K: embed dim
    N = 25000 is a good approximation in general
    """

    S = torch.linalg.svdvals(Z) # singular values
    S_norm1 = torch.linalg.norm(S, 1)

    p = S/S_norm1 + 1e-7 # normalize sum to 1
    entropy = - torch.sum(p*torch.log(p))
    return torch.exp(entropy)


def main(processed_folder):

    if MULTIMODAL:
        # Multimodal BrainGPT (accelerometer)
        model_mae = prepare_pretrained_brainGPT_multimodal(CHECKPOINT_PATH, MODEL_ARCHITECTURE)
        print('Multimodal BrainGPT model loaded.')
    else:
        # Vanilla BrainGPT
        model_mae = prepare_pretrained_brainGPT(CHECKPOINT_PATH, MODEL_ARCHITECTURE)
        print('Vanilla BrainGPT model loaded.')

    device = torch.device('cuda:0')
    model_mae = model_mae.to(device)


    # list subfolders
    recordings = [f.path for f in os.scandir(processed_folder) if f.is_dir()]

    for folder in recordings:
        print("Processing folder:", folder)
        input_path = os.path.join(folder, 'all.mat')
        output_path = os.path.join(folder, 'latent_features' + APPENDIX)

        try:
            all_data = h5py.File(input_path)['X'] # FIXME
        except:
            # lost files that were recomputed later needed to be opened this way :)
            all_data = scipy.io.loadmat(input_path)['x']
            all_data = np.transpose(all_data, (0, 3, 2, 1))

        # expected shape: (n_samples, 32, 24, 10) .. n_samples = 10*seconds
        print(f"dataset shape: {all_data.shape}, expected shape is (n_samples, 32, 24, 10)")

        with torch.no_grad():
            
            model_mae.eval()
            inputs, latents, targets, sessions = [], [], [], []

            for i in tqdm(range(all_data.shape[0])):
                # print(f'epoch {i+1}/{all_data.shape[0]}, ', end='')
                epoch_wavelet = np.transpose(all_data[i,:,:,:], (1, 0, 2))[:,:32,:]
                samp = torch.from_numpy(epoch_wavelet).to(device, non_blocking=True, dtype=torch.float32).unsqueeze(0)

                with torch.cuda.amp.autocast():
                    if MULTIMODAL:
                        lat = model_mae.brainGPT.transform(samp, mask_ratio=0)
                    else:
                        lat = model_mae.transform(samp, mask_ratio=0)
                    latents.append(lat)

            latents = np.concatenate(latents, axis=0)

        print(f'rank latent: {rankme(torch.from_numpy(latents))}') # ~2x4x10
        print(latents.shape)


        struct_name = 'xLatent'
        savemat(output_path, {struct_name: latents.transpose()})
        print(f'Latent features saved to {output_path}')


# MODEL_ARCHITECTURE = 'mae_neut_base_patch245_1implant' # vanilla BrainGPT
# MODEL_ARCHITECTURE = 'mae_neut_conf_tiny_multimodal_mlp_accelerometer' # accelerometer BrainGPT
MODEL_ARCHITECTURE = 'mae_neut_conf_tiny_multimodal_mlp_delta_DINO' # DINO BrainGPT

# vanilla BrainGPT:
# CHECKPOINT_PATH = '/home/cyberspace007/mpicek/NeuralMAE/pretrained_brainGPT/checkpoint-14_up2001.pth'
# DINO BrainGPT:
CHECKPOINT_PATH = "/media/cyberspace007/T7/tmp/training_logs/neuralmae/checkpoints/MLP_predict_DINO_NOT_frozen_BrainGPT_25mask_small_weight_full_retrain_smaller_MLP/checkpoint-29.pth"

# APPENDIX = '.mat'
APPENDIX = '_DINO_SSL_full_retrain_smaller_MLP_29.mat'
# APPENDIX = '_DINO_1.mat'
MULTIMODAL = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process one day of recordings.")
    parser.add_argument("processed_folder", help="Path to the folder containing individual recordings. The output will be saved there, too.")
    args = parser.parse_args()

    print("Model architecture:", MODEL_ARCHITECTURE)
    print("Checkpoint path:", CHECKPOINT_PATH)
    print("Multimodal:", MULTIMODAL)
    print("Appendix:", APPENDIX)
    main(args.processed_folder)

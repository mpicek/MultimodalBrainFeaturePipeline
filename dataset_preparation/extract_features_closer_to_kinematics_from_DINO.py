import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse

###############################################################################
#                                WARNING
# THIS NETWORK IS COPIED FROM predict_kinematics_from_dino.ipynb NOTEBOOK!!!
# PLEASE MAKE SURE TO UPDATE THE NETWORK TO MATCH THE ONE USED IN YOUR EXPERIMENTS
# (BUT NO CHANGE IS PLANNED)
###############################################################################
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=128):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(input_size, 512)
        self.hidden2 = nn.Linear(512, 128)
        self.output = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(128)
    
    def forward(self, x):
        x = torch.relu(self.batch_norm1(self.hidden1(x)))
        x = self.dropout(x)
        x = torch.relu(self.batch_norm2(self.hidden2(x)))
        # save the output of this layer as features to be returned
        features = x
        x = self.output(x)
        return x, features

def main(dino_folder, output_folder, MLP_checkpoint):
    # Load the trained model
    input_size = 2048
    output_size = 10  # Update this to your actual output size
    model = MLP(input_size, output_size).cuda()
    model.load_state_dict(torch.load(MLP_checkpoint))
    model.eval()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each DINO features file
    for dino_file in tqdm(os.listdir(dino_folder)):
        if dino_file.endswith('_features.npy'):
            dino_path = os.path.join(dino_folder, dino_file)
            output_filename = dino_file.replace('_features.npy', '_dino_close_to_kinematics.npy')
            # skip if the output file already exists
            if os.path.exists(os.path.join(output_folder, output_filename)):
                print(f"Features already extracted in {output_filename}. Skipping...")
                continue
            
            # Load data in chunks to avoid memory issues
            dino_features = np.load(dino_path, mmap_mode='r')
            K = dino_features.shape[0]
            batch_size = 64  # Adjust batch size according to your memory limits
            
            new_features = []
            
            for start in range(0, K, batch_size):
                end = min(start + batch_size, K)
                dino_batch = dino_features[start:end]
                
                # Convert to torch tensor
                dino_batch_tensor = torch.tensor(dino_batch, dtype=torch.float32).cuda()
                
                # Extract new features
                with torch.no_grad():
                    _, batch_features = model(dino_batch_tensor)
                    new_features.append(batch_features.cpu().numpy())
            
            # Concatenate all features and save them
            new_features = np.concatenate(new_features, axis=0)
            output_path = os.path.join(output_folder, output_filename)

            np.save(output_path, new_features)

            print(f'Saved new features to {output_path}')

# MLP_checkpoint = '/home/cyberspace007/mpicek/master_project/experiments/MLP_DINO_to_kinematics.pth'
# dino_folder = '/media/cyberspace007/T7/tmp/dino'  # Set this to your DINO features folder
# output_folder = '/media/cyberspace007/T7/tmp/between_dino_and_kinematics'  # Set this to your desired output folder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess the kinematics signals.")
    parser.add_argument("dino_folder", help="Path to the folder containing DINO features in .npy files.")
    parser.add_argument("output_folder", help="Path to the output folder to save the latent features ('something between DINO and kinematics').")
    parser.add_argument("MLP_checkpoint", help="Path to the checkpoint of the pre-trained MLP model trained to map DINO features to kinematics signals.")
    args = parser.parse_args()
    main(args.dino_folder, args.output_folder, args.MLP_checkpoint)

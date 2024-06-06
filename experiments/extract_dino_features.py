import argparse
import json
import numpy as np
import torch as th
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps
import cv2
import os
from RCDM.guided_diffusion_rcdm.get_ssl_models import get_model
from RCDM.guided_diffusion_rcdm.image_datasets import random_crop_arr, center_crop_arr
from RCDM.guided_diffusion_rcdm.script_util import add_dict_to_argparser, model_and_diffusion_defaults, args_to_dict
from tqdm import tqdm

def exclude_bias_and_norm(p):
	return p.ndim == 1

def pad_to_square(image, padding_color=(0, 255, 0)):
	max_dim = max(image.size)
	delta_w = max_dim - image.size[0]
	delta_h = max_dim - image.size[1]
	padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
	return ImageOps.expand(image, padding, padding_color)

class VideoDataset(Dataset):
	def __init__(self, video_path, resolution, transform=None, random_crop=False, random_flip=True):
		self.cap = cv2.VideoCapture(video_path)
		self.resolution = resolution
		self.transform = transform
		self.random_crop = random_crop
		self.random_flip = random_flip
		self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
		print("Frames: ", self.frames)

	def __len__(self):
		return self.frames

	def __getitem__(self, idx):
		if not self.cap.isOpened():
			raise IndexError
		ret, frame = self.cap.read()
		if not ret:
			raise IndexError
		pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		
		# Pad image to make it square
		pil_image = pad_to_square(pil_image)

		if self.random_crop:
			arr = random_crop_arr(pil_image, self.resolution)
			arr2 = random_crop_arr(pil_image, 224)
		else:
			arr = center_crop_arr(pil_image, self.resolution)
			arr2 = center_crop_arr(pil_image, 224)
		if self.random_flip and np.random.rand() < 0.5:
			arr = arr[:, ::-1]
			arr2 = arr2[:, ::-1]
		arr = arr.astype(np.float32) / 127.5 - 1
		arr2 = arr2.astype(np.float32) / 127.5 - 1

		arr = np.transpose(arr, [2, 0, 1])
		arr2 = np.transpose(arr2, [2, 0, 1])
		
		return arr, arr2
	
	def close_video(self):
		self.cap.release()

def main(args):
	args.gpu = 0
	with open('args.json', 'w') as f:
		json.dump(vars(args), f)

	ssl_model = get_model(args.type_model, args.use_head).cuda().eval()
	for p in ssl_model.parameters():
		ssl_model.requires_grad = False

	# list all files ending with segmented_pose.mp4
	segmented_videos = [os.path.join(args.segmented_videos_path, f) for f in os.listdir(args.segmented_videos_path) if f.endswith('segmented_pose.mp4')]
	
	for segmented_video in tqdm(segmented_videos):
		# skip the video if the features are already extracted
		output_features_path = os.path.join(args.output_path, os.path.basename(segmented_video).split('_segmented_pose.mp4')[0] + '_features.npy')
		print(segmented_video)
		if os.path.exists(output_features_path):
			print("Features already extracted. Skipping...")
			continue

		video_dataset = VideoDataset(segmented_video, args.image_size, random_crop=False, random_flip=False)
		data_loader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
		all_features = []
		for batch_small, batch in tqdm(data_loader):
			batch = batch.cuda()
			with th.no_grad():
				feat = ssl_model(batch).detach().cpu().numpy()
			all_features.append(feat[0])
			
		video_dataset.close_video()
		all_features = np.stack(all_features)

		np.save(output_features_path, all_features)

def create_argparser():
	defaults = dict(
		input_directory="/media/cyberspace007/T7/martin/data/pose_segmentation_lower_threshold/",
		image_size=224,
		out_dir=".",
		use_head=False,
	)
	parser = argparse.ArgumentParser()
	add_dict_to_argparser(parser, defaults)
	# input directory
	parser.add_argument('segmented_videos_path', type=str, help='Path to the folder with segmented videos.')
	parser.add_argument('output_path', type=str, help='Path to save the features.')
	parser.add_argument('--type_model', type=str, default="dino", help='Select the type of model to use.')
	return parser

if __name__ == "__main__":
	args = create_argparser().parse_args()
	main(args)

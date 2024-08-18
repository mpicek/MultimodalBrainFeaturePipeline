import argparse
import json
import numpy as np
import torch as th
import os
import time
import cv2  # Import OpenCV for video handling
from torchvision.utils import save_image
from PIL import Image  # Import PIL for image handling

from RCDM.guided_diffusion_rcdm import dist_util, logger
from RCDM.guided_diffusion_rcdm.get_rcdm_models import get_dict_rcdm_model
from RCDM.guided_diffusion_rcdm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def exclude_bias_and_norm(p):
    return p.ndim == 1

def main(args):
    args.gpu = 0

    print("Load features...")
    features = np.load(args.features_path)
    print(features.shape)

    # Open the video file
    cap = cv2.VideoCapture(args.video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {args.video_path}")

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Print the total number of frames and the number of features
    print(f'Total frames in video: {total_frames}')
    print(f'Total features in .npy file: {len(features)}')

    # Check if the number of features matches the number of frames
    if total_frames != len(features):
        raise ValueError("The number of frames in the video and the number of features do not match.")
    else:
        print("The number of frames and features match.")

    if args.subtract_mean and args.subtract_from_frame is not None:
        raise ValueError("Only one of subtract_mean and subtract_from_frame can be True.")

    if args.subtract_mean:
        original_features = np.copy(features)
        # subtract the mean of the whole video
        features -= np.mean(features, axis=0)
        # scale it so that the feature for each frame has norm as the mean norm and it is in the same bounds as the original features
        features *= np.linalg.norm(original_features, axis=1).mean() / np.linalg.norm(features, axis=1).mean()
        # also keep it in the range of min and max (observed from features)
        features = np.clip(features, np.min(original_features), np.max(original_features))

        # compare stats about features and original features
        print("Original features:")
        print("Mean: ", np.mean(original_features, axis=0))
        print("Norm: ", np.linalg.norm(original_features, axis=1).mean())
        print("Norm std", np.linalg.norm(original_features, axis=1).std())
        print("Min: ", np.min(original_features))
        print("Max: ", np.max(original_features))
        print("Features:")
        print("Mean: ", np.mean(features, axis=0))
        print("Norm: ", np.linalg.norm(features, axis=1).mean())
        print("Norm std", np.linalg.norm(features, axis=1).std())
        print("Min: ", np.min(features))
        print("Max: ", np.max(features))

    if args.subtract_from_frame is not None:
        original_features = np.copy(features)
        # subtract the specific feature from all the feature
        features -= features[args.subtract_from_frame]
        # scale it so that the feature for each frame has norm as the mean norm and it is in the same bounds as the original features
        features *= np.linalg.norm(original_features, axis=1).mean() / np.linalg.norm(features, axis=1).mean()
        # also keep it in the range of min and max (observed from features)
        features = np.clip(features, np.min(original_features), np.max(original_features))

    if args.index >= len(features):
        raise IndexError(f"Index {args.index} out of range for features of length {len(features)}")
    
    feat = th.tensor(features[args.index]).unsqueeze(0).cuda()
    
    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()), G_shared=args.no_shared, feat_cond=True, ssl_dim=feat.shape[1]
    )

    print("loading model...")
    if args.model_path == "":
        trained_model = get_dict_rcdm_model(args.type_model, args.use_head)
    else:
        trained_model = th.load(args.model_path, map_location="cpu")
    model.load_state_dict(trained_model, strict=True)
    model.to(dist_util.dev())
    model.eval()

    print("sampling...")
    sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)

    model_kwargs = {"feat": feat}
    
    with th.no_grad():
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    # Set the video to the frame corresponding to the index
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.index)

    # Read the frame
    ret, frame = cap.read()

    if ret:
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame and samples to PIL images
        frame_img = Image.fromarray(frame_rgb)
        generated_imgs = [Image.fromarray(img) for img in sample]

        # Resize generated images to maintain aspect ratio with matching height
        generated_imgs = [
            img.resize(
                (int(frame_img.height * img.width / img.height), frame_img.height),
                Image.BICUBIC
            ) for img in generated_imgs
        ]

        # Determine the size of the final image (width is sum of all images' widths)
        total_width = frame_img.width + sum(img.width for img in generated_imgs)
        max_height = frame_img.height  # All images now have the same height

        # Create a new blank image to combine them
        combined_img = Image.new('RGB', (total_width, max_height))

        # Paste the frame and generated images side by side
        x_offset = 0
        combined_img.paste(frame_img, (x_offset, 0))
        x_offset += frame_img.width

        for img in generated_imgs:
            combined_img.paste(img, (x_offset, 0))
            x_offset += img.width

        # Save the combined image
        output_path = os.path.join(args.out_dir, f"sample_{args.index}.png")
        combined_img.save(output_path)
        print(f"Combined image saved to {output_path}")
    else:
        print(f"Failed to read frame at index {args.index}")

    # Release the video capture object
    cap.release()

def create_argparser():
    defaults = dict(
        data_dir="/home/cyberspace007/Downloads/data",
        clip_denoised=True,
        num_images=4,
        use_ddim=False,
        model_path="",
        submitit=False,
        local_rank=0,
        dist_url="env://",
        G_shared=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--name', default="samples", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--out_dir', default="./samples/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--no_shared', action='store_false', default=True,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--use_head', action='store_true', default=False,
                        help='Use the projector/head to compute the SSL representation instead of the backbone.')
    parser.add_argument('--type_model', type=str, default="dino",
                        help='Select the type of model to use.')
    parser.add_argument('--features_path', type=str, required=True, help='Path to the numpy file containing features.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file corresponding to the features.')
    parser.add_argument('--index', type=int, required=True, help='Index of the feature to use for sampling.')
    parser.add_argument('--subtract_mean', action='store_true', default=False, help='If True, subtract the mean (of the whole video) from the features.')
    parser.add_argument('--subtract_from_frame', type=int, default=None, help='If not None, subtract the feature at this index from all the features.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for sampling.')
    return parser

if __name__ == "__main__":
    args = create_argparser().parse_args()
    t0 = time.time()
    main(args)
    elapsed = time.time() - t0
    print(elapsed)

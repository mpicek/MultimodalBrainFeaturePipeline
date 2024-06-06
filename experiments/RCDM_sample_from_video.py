import argparse
import json
import numpy as np
import torch as th
import os
from torchvision.utils import save_image
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

        # put mean in the features (but broadcast it so that it is of the same shape as features were)

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
            (1, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    # if args.out_dir doesn't exist, make it
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    output_path = os.path.join(args.out_dir, f"sample_{args.index}.png")
    save_image(th.FloatTensor(sample).permute(0, 3, 1, 2), output_path, normalize=True, scale_each=True, nrow=1)
    print(f"Sample saved to {output_path}")

def create_argparser():
    defaults = dict(
        data_dir="/home/cyberspace007/Downloads/data",
        clip_denoised=True,
        num_images=4,
        batch_size=2,
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
    parser.add_argument('--index', type=int, required=True, help='Index of the feature to use for sampling.')
    parser.add_argument('--subtract_mean', action='store_true', default=False, help='If True, subtract the mean (of the whole video) from the features.')
    return parser

if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)

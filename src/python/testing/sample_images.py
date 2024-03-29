""" Script to generate sample images from the diffusion model.

In the generation of the images, the script is using a DDIM scheduler.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from generative.networks.nets import VQVAE, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", help="Path to save the .pth file of the diffusion model.")
    parser.add_argument("--stage1_path", help="Path to the .pth model from the stage1.")
    parser.add_argument("--diffusion_path", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--stage1_config_file_path", help="Path to the .pth model from the stage1.")
    parser.add_argument("--diffusion_config_file_path", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--start_seed", type=int, help="Path to the MLFlow artifact of the stage1.")
    parser.add_argument("--stop_seed", type=int, help="Path to the MLFlow artifact of the stage1.")
    parser.add_argument("--x_size", type=int, default=64, help="Latent space x size.")
    parser.add_argument("--y_size", type=int, default=64, help="Latent space y size.")
    parser.add_argument("--scale_factor", type=float, help="Latent space y size.")
    parser.add_argument("--num_inference_steps", type=int, help="")

    args = parser.parse_args()
    return args


def main(args):
    print_config()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda")

    config = OmegaConf.load(args.stage1_config_file_path)
    stage1 = VQVAE(**config["stage1"]["params"])
    stage1.load_state_dict(torch.load(args.stage1_path))
    stage1.to(device)
    stage1.eval()

    config = OmegaConf.load(args.diffusion_config_file_path)
    diffusion = DiffusionModelUNet(**config["ldm"].get("params", dict()))
    diffusion.load_state_dict(torch.load(args.diffusion_path))
    diffusion.to(device)
    diffusion.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=config["ldm"]["scheduler"]["num_train_timesteps"],
        beta_start=config["ldm"]["scheduler"]["beta_start"],
        beta_end=config["ldm"]["scheduler"]["beta_end"],
        schedule=config["ldm"]["scheduler"]["schedule"],
        prediction_type=config["ldm"]["scheduler"]["prediction_type"],
        clip_sample=False,
    )
    scheduler.set_timesteps(args.num_inference_steps)

    for i in range(args.start_seed, args.stop_seed):
        set_determinism(seed=i)
        noise = torch.randn((1, config["ldm"]["params"]["in_channels"], args.x_size, args.y_size)).to(device)

        with torch.no_grad():
            progress_bar = tqdm(scheduler.timesteps)
            for t in progress_bar:
                noise_pred = diffusion(noise, timesteps=torch.Tensor((t,)).to(noise.device).long())
                noise, _ = scheduler.step(noise_pred, t, noise)

        with torch.no_grad():
            sample = stage1.decode_stage_2_outputs(noise / args.scale_factor)

        sample = np.clip(sample.cpu().numpy(), 0, 1)
        sample = (sample * 255).astype(np.uint8)
        im = Image.fromarray(sample[0, 0])
        im.save(output_dir / f"sample_{i}.jpg")


if __name__ == "__main__":
    args = parse_args()
    main(args)

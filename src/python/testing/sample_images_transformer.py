""" Script to generate sample images from the transformer model."""

import argparse
from pathlib import Path

import numpy as np
import torch
from generative.inferers import VQVAETransformerInferer
from generative.networks.nets import VQVAE, DecoderOnlyTransformer
from generative.utils.enums import OrderingType
from generative.utils.ordering import Ordering
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", help="Path to save the .pth file of the diffusion model.")
    parser.add_argument("--stage1_path", help="Path to the .pth model from the stage1.")
    parser.add_argument("--transformer_path", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--stage1_config_file_path", help="Path to the .pth model from the stage1.")
    parser.add_argument("--transformer_config_file_path", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--start_seed", type=int, help="Path to the MLFlow artifact of the stage1.")
    parser.add_argument("--stop_seed", type=int, help="Path to the MLFlow artifact of the stage1.")

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

    config = OmegaConf.load(args.transformer_config_file_path)
    transformer = DecoderOnlyTransformer(**config["transformer"].get("params", dict()))
    transformer.load_state_dict(torch.load(args.transformer_path))
    transformer.to(device)
    transformer.eval()

    ordering = Ordering(ordering_type=OrderingType.RASTER_SCAN.value, spatial_dims=2, dimensions=(1, 64, 64))

    inferer = VQVAETransformerInferer()

    for i in range(args.start_seed, args.stop_seed):
        set_determinism(seed=i)

        with torch.no_grad():
            sample = inferer.sample(
                vqvae_model=stage1,
                transformer_model=transformer,
                ordering=ordering,
                latent_spatial_dim=(64, 64),
                starting_tokens=stage1.num_embeddings * torch.ones((1, 1), device=device),
            )

        sample = np.clip(sample.cpu().numpy(), 0, 1)
        sample = (sample * 255).astype(np.uint8)
        im = Image.fromarray(sample[0, 0])
        im.save(output_dir / f"sample_{i}.jpg")


if __name__ == "__main__":
    args = parse_args()
    main(args)

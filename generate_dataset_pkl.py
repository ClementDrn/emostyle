import os
import pickle
from typing import List
import argparse

import numpy as np
import PIL.Image
import torch
from torch.nn.functional import batch_norm
import dnnlib
from tqdm import tqdm


def generate_data(
    checkpoint_path: str,
    # stylegan_size: int,
    truncation_psi: float,
    n_samples: int,
    batch_size: int,
    outdir: str,
    on_cpu: bool
):
    device = 'cuda' if not on_cpu and torch.cuda.is_available() else 'cpu'

    with dnnlib.util.open_url(checkpoint_path) as f:
        G = pickle.load(f)['G_ema'].to(device).eval().float()

    mapping = G.mapping.requires_grad_(False)
    synthesis = G.synthesis.requires_grad_(False)

    image_id = 0 
    for i in tqdm(range(image_id, n_samples, batch_size)):
        z = np.random.RandomState().randn(batch_size, 512).astype("float32")
        with torch.no_grad():
            latents = mapping(torch.from_numpy(z).to(device), None, truncation_psi=truncation_psi)
            all_images = synthesis(
                latents,
                noise_mode='const',
                force_fp32=True
            )

            all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

            
            for id in range(all_images.shape[0]):
                PIL.Image.fromarray(all_images[id], 'RGB').save(f'{outdir}/{image_id:06}.png')
                np.save((f'{outdir}/{image_id:06}.npy'), latents[id].cpu().numpy(), allow_pickle=False)
                image_id += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train script for your program")

    parser.add_argument(
        "--outdir", 
        type=str, 
        default="dataset/1024_pkl", 
        help="Output directory")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="pretrained/ffhq.pkl", 
        help="Path to the StyleGAN2 checkpoint")
    # parser.add_argument(
    #     "--size", 
    #     type=int, 
    #     default=1024, 
    #     help="Image size")
    parser.add_argument(
        "--truncation_psi", 
        type=float,
        default=0.7, 
        help="Truncation psi")
    parser.add_argument(
        "--samples", 
        type=int, 
        default=1000, 
        help="Number of samples to generate (must be divisible by batch size)")
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Batch size for generation")
    parser.add_argument(
        "--cpu", 
        action="store_true", 
        help="Run on CPU (if provided)")
    
    args = parser.parse_args()

    # Check if provided paths exist
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file '{args.checkpoint}' does not exist!")
    
    # Check that the number of samples is divisible by the batch size
    if args.samples % args.batch_size != 0:
        raise ValueError(f"Number of samples '{args.samples}' must be divisible by batch size '{args.batch_size}'!")

    # Create output directory if it doesn't exist
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    generate_data(
        checkpoint_path=args.checkpoint,
        # stylegan_size=args.size,
        truncation_psi=args.truncation_psi,
        n_samples=args.samples,
        batch_size=args.batch_size,
        outdir=args.outdir,
        on_cpu=args.cpu
    )

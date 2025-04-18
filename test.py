import os
import pickle
import random
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing_extensions import final
from PIL import Image
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable, grad
from torch.nn.functional import binary_cross_entropy_with_logits
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
from lpips import LPIPS

import pytorch_msssim
from dataset import SyntheticDataset
from models.emo_mapping import EmoMappingW, EmoMappingWplus
from models.landmark import FaceAlignment
from models.emonet import EmoNet
from models.vggface2 import VGGFace2
from models.stylegan2_interface import StyleGAN2
from losses import *
from skimage import draw
import glob


EMO_EMBED = 64
STG_EMBED = 512
INPUT_SIZE = 1024
ITER_NUM = 500

def test(
    images_path: str,
    stylegan2_checkpoint_path: str,
    checkpoint_path: str,
    output_path: str,
    test_mode: str,
    valence: list,
    arousal: list,
    wplus: bool,
    on_cpu: bool
):
    # Set device to CPU if asked, otherwise try to use GPU
    device = 'cuda' if not on_cpu and torch.cuda.is_available() else 'cpu'

    stylegan = StyleGAN2(
        checkpoint_path=stylegan2_checkpoint_path,
        stylegan_size=INPUT_SIZE,
        is_dnn=True,
        is_pkl=True
    )
    stylegan.eval()
    stylegan.requires_grad_(False)  

    ckpt_emo = torch.load("pretrained/emonet_8.pth")
    ckpt_emo = { k.replace('module.',''): v for k,v in ckpt_emo.items() }
    emonet = EmoNet(n_expression=8)
    emonet.load_state_dict(ckpt_emo, strict=False)
    emonet.eval()
    
    ckpt_emo_mapping = torch.load(checkpoint_path, map_location=device)
    if wplus:
        emo_mapping = EmoMappingWplus(INPUT_SIZE, EMO_EMBED, STG_EMBED)
    else:
        emo_mapping = EmoMappingW(EMO_EMBED, STG_EMBED)
    emo_mapping.load_state_dict(ckpt_emo_mapping['emo_mapping_state_dict'])
    emo_mapping.eval()

    emo_mapping = emo_mapping.to(device)
    stylegan = stylegan.to(device)
    emonet = emonet.to(device)
    
    latents = {}
    
    if test_mode == 'random':
        # random mode:
        # - Randomly samples 100 image indices from 1 to 70000.
        # - Saves the random sample indices for reproducibility.
        # - Loads the latent files and corresponding image file paths for each sampled index.
        random_images = random.sample(range(1, 70000), 100)
        with open('random.pkl', 'wb') as f:
            pickle.dump(random_images, f)
    
        for image_number in range(len(random_images)): 
            latent_path = images_path + str(random_images[image_number]).zfill(6) + ".npy" 
            image_path = images_path + str(random_images[image_number]).zfill(6) + ".png"
            image_latent = np.load(latent_path, allow_pickle=False)
            if wplus:
                # For W+ latent space, use the entire latent (all channels)
                image_latent = np.expand_dims(image_latent[:, :], 0)
            else:
                # For standard latent space, use the first latent vector
                image_latent = np.expand_dims(image_latent[0, :], 0)
            image_latent = torch.from_numpy(image_latent).float().to(device)
            latents[image_path] = image_latent
            
    elif test_mode == 'folder_images':
        # folder_images mode:
        # - Iterates over all PNG files in the specified images folder.
        # - For each image, loads the corresponding latent code from a .npy file.
        for image_file in [images_path+filename for filename in os.listdir(images_path) if filename.endswith('.png')]:
            latent_path = os.path.splitext(image_file)[0]+'.npy'
            image_latent = np.load(latent_path, allow_pickle=False)
            if wplus:
                # For W+ latent space, use the entire latent (all channels)
                image_latent = np.expand_dims(image_latent[:, :], 0)
            else:
                # For standard latent space, use the first latent vector
                image_latent = np.expand_dims(image_latent[0, :], 0)
            image_latent = torch.from_numpy(image_latent).float().to(device)
            latents[image_file] = image_latent
        
    num_images = len(valence) * len(arousal)
    for img, latent in latents.items():
        input_image = Image.open(img).convert('RGB') #.transpose(0, 2, 1)
        image_name = os.path.basename(img)

        _ , ax_g = plt.subplots(1, num_images, figsize=(100, 50) )
        if num_images == 1:     # Ensure that ax_g is a list even if there's only one image
            ax_g = [ax_g]
        plt.subplots_adjust(left=.05, right=.95, wspace=0, hspace=0)
        iter = 0
        
        for v_idx in tqdm(range(len(valence))):
            for a_idx in range(len(arousal)):
                emotion = torch.FloatTensor([valence[v_idx], arousal[a_idx]]).float().unsqueeze_(0).to(device)
                fake_latents = latent + emo_mapping(latent, emotion)
                generated_image_tensor = stylegan.generate(fake_latents)
                generated_image_tensor = (generated_image_tensor + 1.) / 2.
                emo_embed = emonet(generated_image_tensor)
                emos = (emo_embed[0][0], emo_embed[0][1])
                
                generated_image = generated_image_tensor.detach().cpu().squeeze().numpy()
                generated_image = np.clip(generated_image*255, 0, 255).astype(np.int32)
                generated_image = generated_image.transpose(1, 2, 0).astype(np.uint8)

                ax_g[iter].imshow(generated_image)
                ax_g[iter].set_title("V: p{:.2f}, r{:.2f}, A: p{:.2f}, r{:.2f}".format(valence[v_idx], emos[0], arousal[a_idx], emos[1]), fontsize=40)
                ax_g[iter].axis('off')
                iter += 1  
        
        plt.savefig("result.png", bbox_inches='tight')
    
        fig = plt.figure(figsize=(80, 20))
        
        output_image = Image.open("result.png").convert('RGB')
        grid = plt.GridSpec(1, num_images+1, wspace=0.05, hspace=0)
        ax_output = fig.add_subplot(grid[0, 1:])
        ax_input = fig.add_subplot(grid[0, 0])
        ax_input.imshow(input_image)
        ax_input.axis('off')
        ax_output.imshow(output_image)
        ax_output.axis('off')
        pos_old = ax_input.get_position()
        pos_new = [pos_old.x0,  0.335,  pos_old.width, pos_old.height]
        ax_input.set_position(pos_new)
        plt.savefig(output_path + "result_{}".format(image_name), bbox_inches='tight')
        plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script")
    
    parser.add_argument(
        "--images_path", 
        type=str, 
        default="dataset/1024_pkl/",
        help="Path to the folder containing images and latents")
    parser.add_argument(
        "--stylegan2_checkpoint_path", 
        type=str, 
        default="pretrained/ffhq.pkl", 
        help="Path to the StyleGAN2 checkpoint")
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default="pretrained/emo_mapping_wplus_2.pt",
        help="Path to the EmoMapping checkpoint")
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="results/",
        help="Path to the output folder")
    parser.add_argument(
        "--test_mode", 
        type=str, 
        default="random", 
        choices=["random", "folder_images"],
        help="Mode of testing: 'random' for random images, 'folder_images' for images in a folder")
    parser.add_argument(
        "--valence", 
        type=float, 
        nargs='+', 
        default=[0.5],
        help="List of valence values to test")
    parser.add_argument(
        "--arousal", 
        type=float, 
        nargs='+', 
        default=[0.5],
        help="List of arousal values to test")
    parser.add_argument(
        "--wplus", 
        action="store_true", 
        help="Use W+ latent space")
    parser.add_argument(
        "--cpu", 
        action="store_true", 
        help="Use CPU instead of GPU")

    args = parser.parse_args()

    # Check if provided paths exist
    if not os.path.exists(args.images_path):
        raise FileNotFoundError(f"Images path '{args.images_path}' does not exist!")
    if not os.path.exists(args.stylegan2_checkpoint_path):
        raise FileNotFoundError(f"StyleGAN2 checkpoint path '{args.stylegan2_checkpoint_path}' does not exist!")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"EmoMapping checkpoint path '{args.checkpoint_path}' does not exist!")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        print(f"Output path '{args.output_path}' created.")

    test(
        images_path=args.images_path,
        stylegan2_checkpoint_path=args.stylegan2_checkpoint_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        test_mode=args.test_mode,
        valence=args.valence,
        arousal=args.arousal,
        wplus=args.wplus,
        on_cpu=args.cpu
    )

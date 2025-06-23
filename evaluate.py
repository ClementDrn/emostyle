import os
import glob
import argparse
import csv
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from models.emonet import EmoNet


def load_emonet(device):
    # Load and prepare EmoNet (assuming similar to test.py usage)
    ckpt_path = "pretrained/emonet_8.pth"
    ckpt_emo = torch.load(ckpt_path, map_location=device)
    # Remove "module." if present.
    ckpt_emo = {k.replace('module.', ''): v for k, v in ckpt_emo.items()}
    model = EmoNet(n_expression=8)
    model.load_state_dict(ckpt_emo, strict=False)
    model.eval()
    model.to(device)

    return model


def evaluate_quantitatively(
    input_dir,
    output_csv=None,
    append=False,
    cpu=False
):
    # Use CPU if asked to or if CUDA is not available
    device = 'cuda' if not cpu and torch.cuda.is_available() else 'cpu'
    
    # Load pretrained EmoNet
    emonet = load_emonet(device)
    
    # Define a basic transform to convert images to tensor in [0,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Get list of images from input directory
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    if not image_files:
        raise FileNotFoundError(f"No PNG files found in '{input_dir}'")
    
    results = []
    for image_file in tqdm(sorted(image_files), desc="Evaluating images"):
        # Fetch image info from filename
        # Expected format: {imagename}_{angle}_{strength}.png or {imagename}_va_{valence}_{arousal}.png
        basename = os.path.basename(image_file)
        name_no_ext = os.path.splitext(basename)[0]
        tokens = name_no_ext.split('_')
        expected_valence = None
        expected_arousal = None
        expected_angle = None
        expected_strength = None
        is_va_format = None

        if len(tokens) > 3 and tokens[-3] == 'va':
            # Format: {imagename}_va_{valence}_{arousal}.png
            is_va_format = True
            expected_valence = float(tokens[-2])
            expected_arousal = float(tokens[-1])
        elif len(tokens) > 2:
            # Format: {imagename}_{angle}_{strength}.png
            is_va_format = False
            expected_angle = float(tokens[-2])
            expected_strength = float(tokens[-1])
            # Compute expected valence and arousal.
            rad = np.deg2rad(expected_angle)
            expected_valence = np.cos(rad) * expected_strength
            expected_arousal = np.sin(rad) * expected_strength
        else:
            print(f"Skipping file {basename}: unexpected format")
            continue
        
        # Load image and convert to tensor.
        try:
            image = Image.open(image_file).convert('RGB')
        except Exception as e:
            print(f"Skipping file {basename}: unable to open image")
            continue
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emo_embed = emonet(tensor)
        # Assume EmoNet output: first two values correspond to valence and arousal.
        predicted_valence = emo_embed[0, 0].item()
        predicted_arousal = emo_embed[0, 1].item()
        
        # Signed errors
        error_valence = predicted_valence - expected_valence
        error_arousal = predicted_arousal - expected_arousal
        
        results.append((basename, expected_valence, expected_arousal, predicted_valence, predicted_arousal, error_valence, error_arousal))
    
    if not results:
        print("No valid evaluations were performed.")
        return
    
    avg_abs_error_valence = np.mean([abs(r[5]) for r in results])
    avg_abs_error_arousal = np.mean([abs(r[6]) for r in results])
    
    print("\nQuantitative Evaluation Results:")
    print(f"Average absolute error in Valence: {avg_abs_error_valence:.4f}")
    print(f"Average absolute error in Arousal: {avg_abs_error_arousal:.4f}")
    
    if output_csv:
        does_file_already_exist = os.path.exists(output_csv)
        with open(output_csv, 'a' if append else 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header only if the file does not exist or if not appending
            if not does_file_already_exist or not append:
                writer.writerow(["filename", "expected_valence", "expected_arousal", "predicted_valence", "predicted_arousal", "error_valence", "error_arousal"])
            for row in results:
                writer.writerow(row)
        print(f"Detailed results written to {output_csv}")


if __name__ == '__main__':
    # Example usage:
    # python evaluate.py --input_dir results/quantitative/ --output_csv evaluation_results.csv --cpu
    parser = argparse.ArgumentParser(description="Quantitative evaluation of generated images using EmoNet")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="results/quantitative/",
        help="Path to the folder containing generated images (named as {imagename}_{angle}_{strength}.png)")
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default=None,
        help="Optional output CSV file to store per-image evaluation results")
    parser.add_argument(
        "--append", 
        "-a",
        action="store_true", 
        help="Append results to the output CSV file if it exists")
    parser.add_argument(
        "--cpu", 
        action="store_true", 
        help="Use CPU instead of GPU")
    
    args = parser.parse_args()

    # Ensure input directory exists
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory '{args.input_dir}' does not exist.")
    # Ensure output CSV directory exists
    if args.output_csv:
        output_dir = os.path.dirname(args.output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Run evaluation
    evaluate_quantitatively(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        append=args.append,
        cpu=args.cpu
    )

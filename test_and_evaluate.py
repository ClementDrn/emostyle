import os
import glob
import shutil
import argparse

from test import test
from evaluate import evaluate_quantitatively


def main(args):
    # Clean up any existing temporary directories
    tmp_input_dir = os.path.join(args.input_dir, "tmp")
    tmp_output_dir = os.path.join(args.output_dir, "tmp")
    if os.path.exists(tmp_input_dir):
        shutil.rmtree(tmp_input_dir)
    if os.path.exists(tmp_output_dir):
        shutil.rmtree(tmp_output_dir)

    # Create temporary and done folders inside the input directory
    if not os.path.exists(tmp_input_dir):
        os.makedirs(tmp_input_dir)
    done_dir = os.path.join(args.input_dir, "done")
    if not os.path.exists(done_dir):
        os.makedirs(done_dir)
    # Create temporary output folder
    if not os.path.exists(tmp_output_dir):
        os.makedirs(tmp_output_dir)

    # Get list of PNG image files directly under input_dir (do not process subfolders)
    image_files = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    if not image_files:
        print(f"No PNG files found in {args.input_dir}")
        return

    for image_file in image_files:
        print(f"Processing {os.path.basename(image_file)} ...")
        # Copy image and its corresponding latent (.npy) into temporary directory
        shutil.copy(image_file, tmp_input_dir)
        latent_file = os.path.splitext(image_file)[0] + ".npy"
        if os.path.exists(latent_file):
            shutil.copy(latent_file, tmp_input_dir)
        else:
            print(f"Warning: latent file {latent_file} not found.")

        # Run test() on the temporary input folder.
        # test() should generate images into args.output_dir.
        test(
            images_path=tmp_input_dir,
            stylegan2_checkpoint_path=args.stylegan2_checkpoint_path,
            checkpoint_path=args.checkpoint_path,
            output_path=tmp_output_dir,
            test_mode=args.test_mode,
            angles=args.angles,
            angle_labels=args.angle_labels,
            strengths=args.strengths,
            valence=args.valence,
            arousal=args.arousal,
            wplus=args.wplus,
            on_cpu=args.cpu
        )

        # Evaluate the generated images and append results to the CSV file.
        evaluate_quantitatively(
            input_dir=tmp_output_dir,
            output_csv=args.output_csv,
            append=True,
            cpu=args.cpu
        )

        # Move the processed image and its latent file to the done folder.
        shutil.move(image_file, os.path.join(done_dir, os.path.basename(image_file)))
        if os.path.exists(latent_file):
            shutil.move(latent_file, os.path.join(done_dir, os.path.basename(latent_file)))

        # If --clean is not provided, move generated images from tmp_output_dir to args.output_dir.
        if not args.clean:
            for gen_img in glob.glob(os.path.join(tmp_output_dir, "*.png")):
                shutil.move(gen_img, args.output_dir)
        # Otherwise, clear the temporary folder.
        else:
            for gen_img in glob.glob(os.path.join(tmp_output_dir, "*.png")):
                os.remove(gen_img)

        # Clear the temporary input directory.
        for f in os.listdir(tmp_input_dir):
            fp = os.path.join(tmp_input_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)

    print("Processing complete.")

    # Clean up temporary directories
    shutil.rmtree(tmp_input_dir)
    shutil.rmtree(tmp_output_dir)
    print("Temporary directories cleaned up.")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and evaluate generated images per input image.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to folder with input images and corresponding latent files.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to folder where test.py will generate images and evaluation is run on.")
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="CSV file path to append evaluation results.")
    parser.add_argument(
        "--stylegan2_checkpoint_path",
        type=str,
        default="pretrained/ffhq.pkl", 
        help="Path to the StyleGAN2 checkpoint.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="pretrained/emo_mapping_wplus_2.pt",
        help="Path to the EmoMapping checkpoint.")
    parser.add_argument(
        "--test_mode",
        type=str,
        default="emotion_singles",
        choices=["random", "folder_images", "emotion_grid", "emotion_singles", "emotion_transition"],
        help="Mode for test()")
    parser.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=None,
        help="List of angles to use (in degrees).")
    parser.add_argument(
        "--angle_labels",
        type=str,
        nargs="+",
        default=None,
        help="List of labels corresponding to each angle.")
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=None,
        help="List of strengths to scale the emotion vector.")
    parser.add_argument(
        "--valence",
        type=float,
        nargs="+",
        default=[0.5],
        help="List of valence values (for evaluation).")
    parser.add_argument(
        "--arousal",
        type=float,
        nargs="+",
        default=[0.5],
        help="List of arousal values (for evaluation).")
    parser.add_argument(
        "--wplus",
        action="store_true",
        help="Use W+ latent space.")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete generated images in output_dir after evaluation.")

    args = parser.parse_args()
    main(args)
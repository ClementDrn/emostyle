import os
import argparse
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm  # import tqdm for the progress bar


def plot_emotion_distributions(
    input_csv: str, 
    output_dir: str,
    are_angles_separated: bool = False,
    are_strengths_separated: bool = False,
):
    """
    Generates plots from the CSV file containing quantitative evaluation results.
    Each plot shows the circumplex (valence vs. arousal) where:
      - Circles represent the expected (valence, arousal)
      - Triangles represent the predicted (valence, arousal)
    The color of points is determined by the emotion's angle (extracted from the filename).
    
    If are_angles_separated is True, a separate plot is generated for each unique angle.
    If are_strengths_separated is True, a separate plot is generated for each unique strength.
    Otherwise an overall plot is created.
    
    Parameters:
      - input_csv (str): Path to CSV file with columns:
           filename, expected_valence, expected_arousal, predicted_valence, predicted_arousal, error_valence, error_arousal.
           The filename should be in the format {imagename}_{angle}_{strength}.png.
      - output_dir (str): Directory to save the generated plots.
      - are_angles_separated (bool): Whether to generate separate plots by angle.
      - are_strengths_separated (bool): Whether to generate separate plots by strength.
    """
    # Read CSV into dataframe.
    df = pd.read_csv(input_csv)
    
    # Parse filename to add two new columns: 'angle' and 'strength'
    def parse_filename(fname):
        base = os.path.splitext(fname)[0]
        tokens = base.split('_')
        try:
            # Expect last two tokens are angle and strength.
            angle = float(tokens[-2])
            strength = float(tokens[-1])
        except Exception:
            angle = np.nan
            strength = np.nan
        return pd.Series({'angle': angle, 'strength': strength})
    
    df = df.join(df['filename'].apply(parse_filename))
    # Remove rows with missing angle/strength info.
    df = df.dropna(subset=['angle', 'strength'])
    
    # Function to plot a dataframe segment.
    def make_plot(plot_df, title, out_filename, enable_colorbar=True):
        fig, ax = plt.subplots(figsize=(8, 8))
        # Instead of using a colormap solely based on angle,
        # we now calculate the hue from the normalized angle and adjust brightness by strength.
        angle_min = 0.0
        angle_max = 360.0
        strength_min = 0.0
        strength_max = 1.0
        
        # Plot expected as circles and predicted as triangles.
        for idx, row in tqdm(plot_df.iterrows(), total=plot_df.shape[0], desc="Plotting points", leave=False):
            # Normalize angle and strength.
            norm_angle = (row['angle'] - angle_min) / (angle_max - angle_min)
            norm_strength = (row['strength'] - strength_min) / (strength_max - strength_min)
            # Define HSV conversion: hue from norm_angle, saturation fixed (0.8),
            # value from strength scaled between 0.5 and 1.
            hsv = (norm_angle, 0.8, 0.25 + 0.75 * norm_strength)
            color = mcolors.hsv_to_rgb(hsv)
            ax.scatter(
                row['expected_valence'], row['expected_arousal'],
                marker='o', color=color, edgecolor='silver', s=100)
            ax.scatter(
                row['predicted_valence'], row['predicted_arousal'],
                marker='^', color=color, edgecolor='k', s=100)
        
        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.set_title(title)
        ax.grid(True)
        if enable_colorbar:
            # Create a dummy ScalarMappable to add a colorbar (using the angle range only)
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'),
                                       norm=plt.Normalize(angle_min, angle_max))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="Angle (°)")

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        fig.savefig(os.path.join(output_dir, out_filename), bbox_inches='tight')
        plt.close(fig)
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if are_angles_separated and are_strengths_separated:
        unique_angles = np.sort(df['angle'].unique())
        unique_strengths = np.sort(df['strength'].unique())
        total_plots = len(unique_angles) * len(unique_strengths)
        desc = "Generating plots by angle and strength"
        for angle, s in tqdm(product(unique_angles, unique_strengths), total=total_plots, desc=desc, leave=False):
            subset = df[(df['angle'] == angle) & (df['strength'] == s)]
            title = f"Emotion Distribution for Angle = {angle}° and Strength = {s}"
            out_file = f"emotion_distribution_angle_{angle}_strength_{s}.png"
            make_plot(subset, title, out_file, False)
    elif are_angles_separated:
        unique_angles = np.sort(df['angle'].unique())
        for angle in tqdm(unique_angles, desc="Generating plots by angle"):
            subset = df[df['angle'] == angle]
            title = f"Emotion Distribution for Angle = {angle}°"
            out_file = f"emotion_distribution_angle_{angle}.png"
            make_plot(subset, title, out_file, False)
    elif are_strengths_separated:
        unique_strengths = np.sort(df['strength'].unique())
        for s in tqdm(unique_strengths, desc="Generating plots by strength"):
            subset = df[df['strength'] == s]
            title = f"Emotion Distribution for Strength = {s}"
            out_file = f"emotion_distribution_strength_{s}.png"
            make_plot(subset, title, out_file, False)
    else:
        for _ in tqdm(range(1), desc="Generating overall plot"):
            title = "Overall Emotion Distribution"
            out_file = "emotion_distribution_overall.png"
            make_plot(df, title, out_file, False)


def plot_error_bars(
    summary_csv: str,
    output_dir: str,
    separate_angles: bool = False,
    separate_strengths: bool = False,
    compare_angles: bool = False,
    compare_strengths: bool = False,
):
    """
    Bar plots of Mean Angle/Strength/Distance Error
    read from summary_csv (must have columns:
      Group Type, Expected Angle, Expected Strength,
      Mean Angle Error, Mean Strength Error, Mean Distance Error).

    The Y axis shows normalized errors (0..1) for each metric.
    Actual error values are shown with units on top of each bar.
    There is a color legend for the three error metrics on the right.

    If compare_angles is True,
      all angles are plotted on the same bar-chart as separate bars,
      multiplying the number of bars by len(angles).
    If compare_strengths is True,
      all strengths are plotted on the same bar-chart as separate bars,
      multiplying the number of bars by len(strengths).

    If separate_angles is True (and compare_angles is False), 
      one bar-chart is generated per unique angle 
      (using the 'angle' group).
    If separate_strengths is True (and compare_strengths is False),
      one bar-chart is generated per unique strength 
      (using the 'strength' group).
    If both 'separate' parameters are True (and both 'compare' parameters are False),
      one bar-chart is generated for each (angle, strength) pair 
      (using the 'angle_strength' group).
    Otherwise, a single bar-chart is created for the overall group.

    Note:
    - 'compare' parameters override their respective 'separate' parameter.
    - A True separate_strengths with a True compare_angles is compatible,
        as well as a True separate_angles with a True compare_strengths.
    - If both 'compare' parameters are True, behavior is undefined.
    """
    # load summary
    df = pd.read_csv(summary_csv)
    df["Expected Angle"] = pd.to_numeric(df["Expected Angle"], errors="coerce")
    df["Expected Strength"] = pd.to_numeric(df["Expected Strength"], errors="coerce")

    os.makedirs(output_dir, exist_ok=True)

    # max‐error normalizers
    norm = {
      "Mean Angle Error": 180.0,
      "Mean Strength Error": 1.0,
      "Mean Distance Error": 2.0
    }
    metrics = list(norm.keys())
    labels = [
        "Mean Angle Err [0, 180]",
        "Mean Strength Err [0, 1]",
        "Mean Distance Err [0, 2]"
    ]
    short_labels = [
        "Angle Err",
        "Strength Err",
        "Distance Err"
    ]
    colors = ["C0", "C1", "C2"]

    def make_chart(row, title, fname):
        heights = [row[m]/norm[m] for m in metrics]
        vals    = [row[m]      for m in metrics]
        fig, ax = plt.subplots(figsize=(4,3))
        bars = ax.bar(short_labels, heights, color=colors)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Normalized Error")
        ax.set_title(title)
        # Values on top of bars
        for idx, (b, v) in enumerate(zip(bars, vals)):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.02,
                    f"{v:.2f}{'°' if 'Angle' in short_labels[idx] else ''}",
                    ha="center", va="bottom", fontsize=8)
        fig.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
        plt.close(fig)

    # Compare all angles in one chart per metric
    if compare_angles:
        if separate_strengths:
            # For each strength, plot all angles together
            strengths = sorted(
                df[df["Group Type"] == "angle_strength"]["Expected Strength"]
                  .astype(float)
                  .unique()
            )
            for s in strengths:
                sub_df = df[
                    (df["Group Type"] == "angle_strength")
                    & (df["Expected Strength"] == s)
                ]
                angles = sub_df["Expected Angle"].astype(float).values
                angle_labels = [f"{int(a)}°" for a in angles]
                x = np.arange(len(angles))
                width = 0.25

                fig, ax = plt.subplots(figsize=(max(8, len(angles) * 0.5), 4))
                for idx, metric in enumerate(metrics):
                    vals = (sub_df[metric].astype(float) / norm[metric]).values
                    ax.bar(x + idx * width, vals, width, label=labels[idx], color=colors[idx])

                ax.set_xticks(x + width)
                ax.set_xticklabels(angle_labels, rotation=90)
                ax.set_ylim(0, 1.0)
                ax.set_ylabel("Normalized Error")
                ax.set_title(f"Normalized Errors by Angle @ Strength={s}")
                ax.legend(loc="upper right")

                fig.tight_layout()
                fig.savefig(
                    os.path.join(output_dir, f"error_compare_angles_strength_{s}.png"),
                    bbox_inches="tight"
                )
                plt.close(fig)
            return
        
        # Basic compare_angles behavior
        angle_df = df[df["Group Type"] == "angle"]
        angles = angle_df["Expected Angle"].astype(float).values
        angle_labels = [f"{int(a)}°" for a in angles]
        x = np.arange(len(angles))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(8, len(angles) * 0.5), 4))
        for idx, metric in enumerate(metrics):
            vals = (angle_df[metric].astype(float) / norm[metric]).values
            ax.bar(x + idx * width, vals, width, label=labels[idx], color=colors[idx])

        ax.set_xticks(x + width)
        ax.set_xticklabels(angle_labels, rotation=90)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Normalized Error")
        ax.set_title("Normalized Errors by Angle")
        ax.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "error_compare_angles.png"),
            bbox_inches="tight"
        )
        plt.close(fig)
        return

    # Compare all strengths in one chart per metric
    if compare_strengths:
        if separate_angles:
            # For each angle, plot all strengths together
            angles = sorted(
                df[df["Group Type"] == "angle_strength"]["Expected Angle"]
                  .astype(float)
                  .unique()
            )
            for a in angles:
                sub_df = df[
                    (df["Group Type"] == "angle_strength")
                    & (df["Expected Angle"] == a)
                ]
                strengths = sub_df["Expected Strength"].astype(float).values
                strength_labels = [f"{float(s)}" for s in strengths]
                x = np.arange(len(strengths))
                width = 0.25

                fig, ax = plt.subplots(figsize=(max(8, len(strengths) * 0.5), 4))
                for idx, metric in enumerate(metrics):
                    vals = (sub_df[metric].astype(float) / norm[metric]).values
                    ax.bar(x + idx * width, vals, width, label=labels[idx], color=colors[idx])

                ax.set_xticks(x + width)
                ax.set_xticklabels(strength_labels, rotation=90)
                ax.set_ylim(0, 1.0)
                ax.set_ylabel("Normalized Error")
                ax.set_title(f"Normalized Errors by Strength @ Angle={a}°")
                ax.legend(loc="upper right")

                fig.tight_layout()
                fig.savefig(
                    os.path.join(output_dir, f"error_compare_strengths_angle_{a}.png"),
                    bbox_inches="tight"
                )
                plt.close(fig)
            return
        
        # Basic compare_strengths behavior
        strength_df = df[df["Group Type"] == "strength"]
        strengths = strength_df["Expected Strength"].astype(float).values
        strength_labels = [f"{float(s)}" for s in strengths]
        x = np.arange(len(strengths))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(8, len(strengths) * 0.5), 4))
        for idx, metric in enumerate(metrics):
            vals = (strength_df[metric].astype(float) / norm[metric]).values
            ax.bar(x + idx * width, vals, width, label=labels[idx], color=colors[idx])

        ax.set_xticks(x + width)
        ax.set_xticklabels(strength_labels, rotation=90)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Normalized Error")
        ax.set_title("Normalized Errors by Strength")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "error_compare_strengths.png"),
            bbox_inches="tight"
        )
        plt.close(fig)
        return

    # Per (angle, strength)
    if separate_angles and separate_strengths:
        for a in sorted(df[df["Group Type"]=="angle_strength"]["Expected Angle"].unique()):
            for s in sorted(df[df["Group Type"]=="angle_strength"]["Expected Strength"].unique()):
                sub = df[
                  (df["Group Type"]=="angle_strength") &
                  (df["Expected Angle"]==a) &
                  (df["Expected Strength"]==s)
                ]
                if sub.empty: continue
                row = sub.iloc[0]
                title = f"Errors @ Angle={a},Strength={s}"
                fname = f"errors_angle_{a}_strength_{s}.png"
                make_chart(row, title, fname)

    # Per angle only
    elif separate_angles:
        for a in sorted(df[df["Group Type"]=="angle"]["Expected Angle"].unique()):
            row = df[(df["Group Type"]=="angle")&(df["Expected Angle"]==a)].iloc[0]
            title = f"Errors @ Angle={a}"
            fname = f"errors_angle_{a}.png"
            make_chart(row, title, fname)

    # Per strength only
    elif separate_strengths:
        for s in sorted(df[df["Group Type"]=="strength"]["Expected Strength"].unique()):
            row = df[(df["Group Type"]=="strength")&(df["Expected Strength"]==s)].iloc[0]
            title = f"Errors @ Strength={s}"
            fname = f"errors_strength_{s}.png"
            make_chart(row, title, fname)

    # Overall
    else:
        row = df[df["Group Type"]=="overall"].iloc[0]
        make_chart(row, "Overall Mean Errors", "errors_overall.png")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize quantitative evaluation results from CSV and generate emotion distribution plots")
    parser.add_argument(
        "plot_type",
        type=str,
        choices=["emotion_dist", "error_bars"],
        help="Type of plot to generate: 'emotion_dist' for expected and predicted emotion distribution, 'error_bars' for error bar plots")
    parser.add_argument(
        "--evaluation_csv",
        type=str,
        default="evaluation_results.csv",
        help="Path to the CSV file with evaluation results (default: evaluation_results.csv)")
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="summary_statistics.csv",
        help="Path to the summary CSV file (default: summary_statistics.csv)")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="plots/",
        help="Directory to save the generated plots (default: 'plots/')")
    parser.add_argument(
        "--separate_angles",
        action="store_true",
        help="Generate separate plots for each unique angle")
    parser.add_argument(
        "--separate_strengths",
        action="store_true",
        help="Generate separate plots for each unique strength")
    parser.add_argument(
        "--compare_angles",
        action="store_true",
        help="In 'error_bars' mode, display individual angle errors on the same plot")
    parser.add_argument(
        "--compare_strengths",
        action="store_true",
        help="In 'error_bars' mode, display individual strength errors on the same plot")
    
    args = parser.parse_args()

    # Ensure input CSV exists.
    target_csv = args.summary_csv if args.plot_type=="error_bars" else args.input_csv
    if not os.path.exists(target_csv):
        raise FileNotFoundError(f"Input CSV file '{target_csv}' does not exist.")
    # Ensure output directory exists.
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.plot_type == "emotion_dist":
        plot_emotion_distributions(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            are_angles_separated=args.separate_angles,
            are_strengths_separated=args.separate_strengths
        )
    else:  # errors
        plot_error_bars(
            summary_csv=args.summary_csv,
            output_dir=args.output_dir,
            separate_angles=args.separate_angles,
            separate_strengths=args.separate_strengths,
            compare_angles=args.compare_angles,
            compare_strengths=args.compare_strengths
        )

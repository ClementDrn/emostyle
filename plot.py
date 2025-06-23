import os
import argparse
import pandas as pd
import numpy as np
from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from tqdm import tqdm  # import tqdm for the progress bar


def plot_emotion_distributions(
    evaluation_csv: str, 
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
      - evaluation_csv (str): Path to CSV file with columns:
           filename, expected_valence, expected_arousal, predicted_valence, predicted_arousal, error_valence, error_arousal.
           The filename should be in the format {imagename}_{angle}_{strength}.png.
      - output_dir (str): Directory to save the generated plots.
      - are_angles_separated (bool): Whether to generate separate plots by angle.
      - are_strengths_separated (bool): Whether to generate separate plots by strength.
    """
    # Read CSV into dataframe.
    df = pd.read_csv(evaluation_csv)
    
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
                angle_labels = [f"{a:.1f}°" for a in angles]
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
        angle_labels = [f"{a:.1f}°" for a in angles]
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
        

def plot_error_boxes(
    evaluation_csv: str,
    summary_csv: str,  # unused now; kept for compatibility
    output_dir: str,
    separate_angles: bool = False,
    separate_strengths: bool = False,
    compare_angles: bool = False,
    compare_strengths: bool = False
):
    """
    Box plots of Angle/Strength Error
    Read from evaluation_csv, which must have columns:
      filename,expected_valence,expected_arousal,predicted_valence,predicted_arousal
    where filename is in the format {imagename}_{angle}_{strength}.png.

    The Y axis shows:
    - Relative Angle Error (-180..180) as boxes
    - Relative Strength Error (-1..1) as boxes

    The X axis shows tested angles or strengths depending on the parameters.

    If compare_angles is True,
      all angles are plotted on the same box-plot as separate boxes,
      multiplying the number of boxes by len(angles).
    If compare_strengths is True,
      all strengths are plotted on the same box-plot as separate boxes,
      multiplying the number of boxes by len(strengths).

    If separate_angles is True (and compare_angles is False),
      one box-plot is generated per unique angle.
    If separate_strengths is True (and compare_strengths is False),
      one box-plot is generated per unique strength.
    If both separate parameters are True (and both compare parameters are False),
      one box-plot is generated for each (angle, strength) pair.
    Otherwise, a single box-plot is created for the overall group.
    """
    # Load CSV.
    df = pd.read_csv(evaluation_csv)

    # Compute predicted angles and strengths from valence/arousal
    df["predicted_angle"] = np.rad2deg(np.arctan2(df["predicted_arousal"], df["predicted_valence"]))
    df["predicted_strength"] = np.sqrt(df["predicted_valence"]**2 + df["predicted_arousal"]**2)

    # Parse filename to add Expected Angle and Expected Strength.
    def parse_filename(fname):
        base = os.path.splitext(fname)[0]
        tokens = base.split('_')
        try:
            angle = float(tokens[-2])
            strength = float(tokens[-1])
        except Exception:
            angle = np.nan
            strength = np.nan
        return pd.Series({"expected_angle": angle, "expected_strength": strength})
    
    df = df.join(df["filename"].apply(parse_filename))
    df["expected_angle"] = pd.to_numeric(df["expected_angle"], errors="coerce")
    df["expected_strength"] = pd.to_numeric(df["expected_strength"], errors="coerce")
    df = df.dropna(subset=["expected_angle", "expected_strength"])
    os.makedirs(output_dir, exist_ok=True)

    # Compute errors
    df["error_angle"] = (df["predicted_angle"] - df["expected_angle"] + 180) % 360 - 180  # Normalize to [-180, 180]
    df["error_strength"] = df["predicted_strength"] - df["expected_strength"]

    # Helper: creates a simple boxplot for a given data array.
    def simple_boxplot(data, title, out_fname, xtick_labels=["Angle Err", "Strength Err"]):
        fig, ax = plt.subplots(figsize=(6,4))
        ax2 = ax.twinx()
        # Define fixed x positions for the two boxes. 
        pos_angle = 0.85
        pos_strength = 1.15
        # Plot angle error on the left axis. 
        boxes_angle = ax.boxplot(data[0], positions=[pos_angle], widths=0.2, patch_artist=True)
        # Plot strength error on the right axis.
        boxes_strength = ax2.boxplot(data[1], positions=[pos_strength], widths=0.2, patch_artist=True)
        
        # Set colors
        for box in boxes_angle['boxes']:
            box.set_facecolor("lightblue")
        for box in boxes_strength['boxes']:
            box.set_facecolor("lightgreen")

        # Set x-axis tick to match the number of boxes.
        ax.set_xticks([pos_angle, pos_strength])
        ax.set_xticklabels(xtick_labels, rotation=90) 

        # Left axis for angle error in degrees.
        ax.set_ylabel("Angle Error (°)")
        ax.set_ylim(-180, 180)
        # Right axis for strength error.
        ax2.set_ylabel("Strength Error")
        ax2.set_ylim(-1, 1)
        # Add horizontal line at y=0 on the left axis.
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        
        ax.set_title(title)
        fig.savefig(os.path.join(output_dir, out_fname), bbox_inches="tight")
        plt.close(fig)

    # Grouped boxplot helper (for compare_ mode).
    def grouped_boxplot(groups, positions, xtick_labels, title, out_fname):
        mpl.rcParams.update({'font.size': max(10, len(groups["Angle"]) * 0.7)})
        fig, ax = plt.subplots(figsize=(max(8, len(xtick_labels)*1.5), max(4, len(groups["Angle"])*0.4)))
        # Create a twin axis.
        ax2 = ax.twinx()
        
        # Plot angle errors on left axis.
        boxes_angle = ax.boxplot(groups["Angle"], positions=positions["Angle"], widths=0.3, patch_artist=True)
        # Plot strength errors on right axis.
        boxes_strength = ax2.boxplot(groups["Strength"], positions=positions["Strength"], widths=0.3, patch_artist=True)
        
        # Set colors.
        for box in boxes_angle['boxes']:
            box.set_facecolor("lightblue")
        for box in boxes_strength['boxes']:
            box.set_facecolor("lightgreen")
        
        # Set x-axis labels.
        ax.set_xticks(np.arange(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels, rotation=90)
        
        # Left axis for angle error in degrees.
        ax.set_ylabel("Angle Error (°)")
        ax.set_ylim(-180, 180)
        # Right axis for strength error.
        ax2.set_ylabel("Strength Error")
        ax2.set_ylim(-1, 1)
        # Add horizontal line at y=0 on the left axis.
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        
        ax.set_title(title)
        fig.savefig(os.path.join(output_dir, out_fname), bbox_inches="tight")
        plt.close(fig)

    # ---- Grouping logic ----
    # compare_angles: group by Expected Angle.
    if compare_angles:
        if separate_strengths:
            # For each unique strength, group by Expected Angle.
            for s in sorted(df["expected_strength"].unique()):
                sub_df = df[df["expected_strength"] == s]
                angles = sorted(sub_df["expected_angle"].unique())
                groups = {"Angle": [], "Strength": []}
                for a in angles:
                    tmp = sub_df[sub_df["expected_angle"] == a]
                    groups["Angle"].append(tmp["error_angle"].dropna().values)
                    groups["Strength"].append(tmp["error_strength"].dropna().values)
                pos = {"Angle": np.arange(len(angles)) - 0.15,
                       "Strength": np.arange(len(angles)) + 0.15}
                title = f"Relative Errors by Angle @ Strength={s}"
                out_fname = f"error_boxes_compare_angles_strength_{s}.png"
                xtick_labels = [f"{a:.1f}°" for a in angles]
                grouped_boxplot(groups, pos, xtick_labels, title, out_fname)
        else:
            # Overall compare angles: group by Expected Angle.
            angles = sorted(df["expected_angle"].unique())
            groups = {"Angle": [], "Strength": []}
            for a in angles:
                tmp = df[df["expected_angle"] == a]
                groups["Angle"].append(tmp["error_angle"].dropna().values)
                groups["Strength"].append(tmp["error_strength"].dropna().values)
            pos = {"Angle": np.arange(len(angles)) - 0.15,
                   "Strength": np.arange(len(angles)) + 0.15}
            title = "Relative Errors by Angle"
            out_fname = "error_boxes_compare_angles.png"
            xtick_labels = [f"{a:.1f}°" for a in angles]
            grouped_boxplot(groups, pos, xtick_labels, title, out_fname)
        return

    # compare_strengths: group by Expected Strength.
    if compare_strengths:
        if separate_angles:
            # For each unique angle, group by Expected Strength.
            for a in sorted(df["expected_angle"].unique()):
                sub_df = df[df["expected_angle"] == a]
                strengths = sorted(sub_df["expected_strength"].unique())
                groups = {"Angle": [], "Strength": []}
                for s in strengths:
                    tmp = sub_df[sub_df["expected_strength"] == s]
                    groups["Angle"].append(tmp["error_angle"].dropna().values)
                    groups["Strength"].append(tmp["error_strength"].dropna().values)
                pos = {"Angle": np.arange(len(strengths)) - 0.15,
                       "Strength": np.arange(len(strengths)) + 0.15}
                title = f"Relative Errors by Strength @ Angle={a}°"
                out_fname = f"error_boxes_compare_strengths_angle_{a}.png"
                xtick_labels = [f"{s}" for s in strengths]
                grouped_boxplot(groups, pos, xtick_labels, title, out_fname)
        else:
            # Overall compare strengths: group by Expected Strength.
            strengths = sorted(df["expected_strength"].unique())
            groups = {"Angle": [], "Strength": []}
            for s in strengths:
                tmp = df[df["expected_strength"] == s]
                groups["Angle"].append(tmp["error_angle"].dropna().values)
                groups["Strength"].append(tmp["error_strength"].dropna().values)
            pos = {"Angle": np.arange(len(strengths)) - 0.15,
                   "Strength": np.arange(len(strengths)) + 0.15}
            title = "Relative Errors by Strength"
            out_fname = "error_boxes_compare_strengths.png"
            xtick_labels = [f"{s}" for s in strengths]
            grouped_boxplot(groups, pos, xtick_labels, title, out_fname)
        return

    # Separate (non-comparison) plots.
    if separate_angles and separate_strengths:
        # One box plot per (angle, strength) pair.
        for a in sorted(df["expected_angle"].unique()):
            for s in sorted(df["expected_strength"].unique()):
                sub = df[(df["expected_angle"] == a) & (df["expected_strength"] == s)]
                if sub.empty: continue
                data = [sub["error_angle"].dropna().values,
                        sub["error_strength"].dropna().values]
                title = f"Relative Errors @ Angle={a}°, Strength={s}"
                out_fname = f"error_boxes_angle_{a}_strength_{s}.png"
                simple_boxplot(data, title, out_fname)
        return

    elif separate_angles:
        # One box plot per angle.
        for a in sorted(df["expected_angle"].unique()):
            sub = df[df["expected_angle"] == a]
            data = [sub["error_angle"].dropna().values,
                    sub["error_strength"].dropna().values]
            title = f"Relative Errors @ Angle={a}°"
            out_fname = f"error_boxes_angle_{a}.png"
            simple_boxplot(data, title, out_fname)
        return

    elif separate_strengths:
        # One box plot per strength.
        for s in sorted(df["expected_strength"].unique()):
            sub = df[df["expected_strength"] == s]
            data = [sub["error_angle"].dropna().values,
                    sub["error_strength"].dropna().values]
            title = f"Relative Errors @ Strength={s}"
            out_fname = f"error_boxes_strength_{s}.png"
            simple_boxplot(data, title, out_fname)
        return

    # Overall plot.
    else:
        data = [df["error_angle"].dropna().values,
                df["error_strength"].dropna().values]
        title = "Overall Relative Errors"
        out_fname = "error_boxes_overall.png"
        simple_boxplot(data, title, out_fname)
        return


def plot_error_heat_map(
    evaluation_csv: str, 
    output_dir: str, 
    steps: int = 10,
    interpolation: str = "quadric",
    component: str = "both"
):
    """
    Generates a heatmap of error(s) computed from evaluation_csv.
    
    For component=="both", an integrated 2D heatmap is generated where:
    - Red channel encodes average valence error
    - Green channel encodes average arousal error
    - Blue is fixed (0.5)
    This produces a 2D color map with integrated legend.
    
    For component=="valence" or "arousal", a single-component heatmap is generated
    using a red or green colormap respectively.
    
    The error normalization is such that 0 error maps to 0.5 (or the middle of the colormap),
    -max maps to 0 and +max maps to 1 for the integrated RGB map.
    
    Parameters:
    - evaluation_csv: CSV file with columns:
        filename, expected_valence, expected_arousal, predicted_valence, 
        predicted_arousal, error_valence, error_arousal
    - output_dir: folder in which to save the plot
    - steps: grid resolution (default=10)
    - interpolation: interpolation method for imshow (default: "quadric")
    - component: one of "both" (default), "valence", or "arousal"
    """
    df = pd.read_csv(evaluation_csv)

    # Determine grid boundaries from expected values.
    x_min, x_max = df["expected_valence"].min(), df["expected_valence"].max()
    y_min, y_max = df["expected_arousal"].min(), df["expected_arousal"].max()

    # Create grid bins.
    x_bins = np.linspace(x_min, x_max, steps+1)
    y_bins = np.linspace(y_min, y_max, steps+1)

    # Prepare grids for average errors.
    heatmap_v = np.full((steps, steps), np.nan)
    heatmap_a = np.full((steps, steps), np.nan)

    # Loop over grid cells.
    for i in range(steps):
        for j in range(steps):
            x_low, x_high = x_bins[i], x_bins[i+1]
            y_low, y_high = y_bins[j], y_bins[j+1]
            subset = df[
                (df["expected_valence"] >= x_low) & (df["expected_valence"] < x_high) &
                (df["expected_arousal"] >= y_low) & (df["expected_arousal"] < y_high)
            ]
            if not subset.empty:
                heatmap_v[j, i] = subset["error_valence"].mean()
                heatmap_a[j, i] = subset["error_arousal"].mean()
            else:
                heatmap_v[j, i] = np.nan
                heatmap_a[j, i] = np.nan

    # Both valence and arousal heatmap
    if component == "both":
        # Normalize each error heatmap to a [0,1] scale using a diverging normalization:
        # 0 error becomes 0.5, positive errors increase toward 1, negative errors decrease toward 0.
        max_abs_v = np.nanmax(np.abs(heatmap_v))
        max_abs_a  = np.nanmax(np.abs(heatmap_a))
        # Avoid division by zero.
        if max_abs_v == 0: 
            max_abs_v = 1
        if max_abs_a == 0:
            max_abs_a = 1

        norm_val = 0.5 + 0.5 * (heatmap_v / max_abs_v)
        norm_ar  = 0.5 + 0.5 * (heatmap_a / max_abs_a)

        # Build an RGB image: red channel encodes valence error, green channel encodes arousal error.
        rgb_image = np.zeros((steps, steps, 3))
        rgb_image[..., 0] = norm_val      # Red channel.
        rgb_image[..., 1] = norm_ar       # Green channel.
        rgb_image[..., 2] = .5            # Blue constant across cells.

        # Plot the RGB image.
        fig, ax = plt.subplots(figsize=(9, 9))
        im = ax.imshow(rgb_image, origin="lower",
                    extent=[x_min, x_max, y_min, y_max],
                    aspect="equal",
                    interpolation="quadric")
        ax.set_xlabel("Expected Valence")
        ax.set_ylabel("Expected Arousal")
        ax.set_title("Average 2D Error Vector Heatmap")

        # Create an integrated 2D colorbar (legend) as an inset axis.
        # This shows the mapping for both valence and arousal errors.
        res = 256
        x_cb = np.linspace(-1, 1, res)
        y_cb = np.linspace(-1, 1, res)
        xx, yy = np.meshgrid(x_cb, y_cb)
        rgb_cb = np.zeros((res, res, 3))
        # Normalize errors using the same formula: 0 error -> 0.5, -1 -> 0, +1 -> 1
        rgb_cb[..., 0] = 0.5 + 0.5 * xx   # Valence error mapping -> Red channel
        rgb_cb[..., 1] = 0.5 + 0.5 * yy   # Arousal error mapping -> Green channel
        rgb_cb[..., 2] = 0.5              # Blue
        
        # Add a new axes for the integrated colorbar.
        ax_cb = fig.add_axes([1.0, 0.4, 0.2, 0.2])  # Adjust position/size as needed.
        ax_cb.imshow(rgb_cb, origin="lower", extent=[-1, 1, -1, 1], aspect="auto")
        ax_cb.set_xlabel("Valence Error", fontsize=10)
        ax_cb.set_ylabel("Arousal Error", fontsize=10)
        ax_cb.set_title("Legend", fontsize=10)
        # Add ticks and labels (unormalized).
        ax_cb.set_xticks([-1, 0, 1])
        ax_cb.set_xticklabels([f"{-max_abs_v:.2f}", "0", f"{max_abs_v:.2f}"], fontsize=10)
        ax_cb.set_yticks([-1, 0, 1])
        ax_cb.set_yticklabels([f"{-max_abs_a:.2f}", "0", f"{max_abs_a:.2f}"], fontsize=10)

    
    # Valence only heatmap
    elif component == "valence":
        # Get max absolute value for normalization.
        max_abs_v = np.nanmax(np.abs(heatmap_v))
        if max_abs_v == 0:
            max_abs_v = 1

        # Custom cmap: red channel varies, green and blue are fixed.
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "valence_error_cmap",
            [(0.0, 0.5, 0.5), (1.0, 0.5, 0.5)], 
            N=256
        )

        # Create the heatmap plot.
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(
            heatmap_v,
            cmap=cmap,
            norm=mcolors.Normalize(vmin=-max_abs_v, vmax=max_abs_v),
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            aspect="equal",
            interpolation=interpolation
        )
        ax.set_xlabel("Expected Valence")
        ax.set_ylabel("Expected Arousal")
        ax.set_title("Average Valence Error Heatmap")
        
        # Add a colorbar legend
        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label("Valence Error", fontsize=10)
        cbar.ax.tick_params(labelsize=10)

    # Arousal only heatmap
    elif component == "arousal":
        # Get max absolute value for normalization.
        max_abs_a = np.nanmax(np.abs(heatmap_a))
        if max_abs_a == 0:
            max_abs_a = 1

        # Custom cmap: green channel varies, red and blue are fixed.
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "arousal_error_cmap",
            [(0.5, 0.0, 0.5), (0.5, 1.0, 0.5)], 
            N=256
        )

        # Create the heatmap plot.
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(
            heatmap_a,
            cmap=cmap,
            norm=mcolors.Normalize(vmin=-max_abs_a, vmax=max_abs_a),
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            aspect="equal",
            interpolation=interpolation
        )
        ax.set_xlabel("Expected Valence")
        ax.set_ylabel("Expected Arousal")
        ax.set_title("Average Arousal Error Heatmap")
        
        # Add a colorbar legend
        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label("Arousal Error", fontsize=10)
        cbar.ax.tick_params(labelsize=10)

    else:
        raise ValueError("Invalid heatmap component. Choose among 'both', 'valence', or 'arousal'.")

    # Save the figure.
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"error_heat_map_{component}.png")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize quantitative evaluation results from CSV and generate emotion distribution plots")
    parser.add_argument(
        "plot_type",
        type=str,
        choices=["emotion_dist", "error_bars", "error_boxes", "error_heat_map"],
        help="Type of plot: 'emotion_dist', 'error_bars', 'error_boxes' or 'error_heat_map'")
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
        help="In 'error_bars' and 'error_boxes' modes, display individual angle errors on the same plot")
    parser.add_argument(
        "--compare_strengths",
        action="store_true",
        help="In 'error_bars' and 'error_boxes' modes, display individual strength errors on the same plot")
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of grid steps for error_heat_map (default: 10).")
    parser.add_argument(
        "--interpolation",
        type=str,
        default="quadric",
        help="Interpolation method for heatmap plotting (e.g. 'quadric', 'nearest')")
    parser.add_argument(
        "--heatmap_component",
        type=str,
        choices=["both", "valence", "arousal"],
        default="both",
        help="For error_heat_map, choose which error component to plot (default: both)")

    args = parser.parse_args()

    # Ensure input CSV exists.
    target_csv = args.evaluation_csv if args.plot_type == "emotion_dist" else args.summary_csv
    if not os.path.exists(target_csv):
        raise FileNotFoundError(f"Input CSV file '{target_csv}' does not exist.")
    # Ensure output directory exists.
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.plot_type == "emotion_dist":
        plot_emotion_distributions(
            evaluation_csv=args.evaluation_csv,
            output_dir=args.output_dir,
            are_angles_separated=args.separate_angles,
            are_strengths_separated=args.separate_strengths
        )
    elif args.plot_type == "error_bars":
        plot_error_bars(
            summary_csv=args.summary_csv,
            output_dir=args.output_dir,
            separate_angles=args.separate_angles,
            separate_strengths=args.separate_strengths,
            compare_angles=args.compare_angles,
            compare_strengths=args.compare_strengths
        )
    elif args.plot_type == "error_boxes":
        plot_error_boxes(
            evaluation_csv=args.evaluation_csv,
            summary_csv=args.summary_csv,
            output_dir=args.output_dir,
            separate_angles=args.separate_angles,
            separate_strengths=args.separate_strengths,
            compare_angles=args.compare_angles,
            compare_strengths=args.compare_strengths
        )
    elif args.plot_type == "error_heat_map":
        plot_error_heat_map(
            evaluation_csv=args.evaluation_csv,
            output_dir=args.output_dir,
            steps=args.steps,
            interpolation=args.interpolation,
            component=args.heatmap_component
        )
    else:
        raise ValueError(f"Unknown plot type: {args.plot_type}. Supported types are 'emotion_dist',"
                         "'error_bars' and 'error_boxes'.")

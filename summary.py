import os
import csv
import argparse
import math
import numpy as np
from collections import defaultdict


def angular_error(pred, exp):
    # Ensure both angles are in [0,360)
    pred = pred % 360
    exp = exp % 360
    diff = abs(pred - exp) % 360
    if diff > 180:
        diff = 360 - diff
    return diff

def compute_group_summary(items, group_type, key_val=None):
    """
    items: list of dict items (each with keys: predicted_angle, predicted_strength, predicted_valence,
           predicted_arousal, angle_error, strength_error, valence_error, arousal_error)
    group_type: one of "angle_strength", "angle", "strength", "overall"
    key_val: for groups not by both fields, the key (angle or strength) to show.
    Returns a dictionary with summary mean values.
    """
    mean_angle = np.mean([it["predicted_angle"] for it in items])
    mean_strength = np.mean([it["predicted_strength"] for it in items])
    mean_valence = np.mean([it["predicted_valence"] for it in items])
    mean_arousal = np.mean([it["predicted_arousal"] for it in items])
    mean_angle_error = np.mean([it["angle_error"] for it in items])
    mean_strength_error = np.mean([it["strength_error"] for it in items])
    mean_valence_error = np.mean([it["valence_error"] for it in items])
    mean_arousal_error = np.mean([it["arousal_error"] for it in items])
    mean_distance_error = np.mean([it["distance_error"] for it in items])
    
    if group_type == "angle_strength":
        expected_angle, expected_strength = key_val
    elif group_type == "angle":
        expected_angle, expected_strength = key_val, "ALL"
    elif group_type == "strength":
        expected_angle, expected_strength = "ALL", key_val
    else: # overall
        expected_angle, expected_strength = "ALL", "ALL"
        
    return {
        "Group Type": group_type,
        "Expected Angle": expected_angle,
        "Expected Strength": expected_strength,
        "Mean Angle": f"{mean_angle:.4f}",
        "Mean Strength": f"{mean_strength:.4f}",
        "Mean Valence": f"{mean_valence:.4f}",
        "Mean Arousal": f"{mean_arousal:.4f}",
        "Mean Angle Error": f"{mean_angle_error:.4f}",
        "Mean Strength Error": f"{mean_strength_error:.4f}",
        "Mean Valence Error": f"{mean_valence_error:.4f}",
        "Mean Arousal Error": f"{mean_arousal_error:.4f}",
        "Mean Distance Error": f"{mean_distance_error:.4f}",
    }

def summarize(input_csv: str, output_csv: str = None):
    # Group data in four ways:
    # 1. By (expected_angle, expected_strength)
    groups_as = defaultdict(list)
    # 2. By expected_angle only
    groups_angle = defaultdict(list)
    # 3. By expected_strength only
    groups_strength = defaultdict(list)
    # 4. Overall
    overall = []
    
    with open(input_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Get errors computed from evaluate.py
                val_err = float(row["error_valence"])
                arr_err = float(row["error_arousal"])
                
                # Expected angle and strength from filename.
                filename = row["filename"]
                base = os.path.splitext(filename)[0]
                tokens = base.split('_')
                if len(tokens) < 2:
                    continue
                expected_angle = float(tokens[-2])
                expected_strength = float(tokens[-1])
                
                # Predicted valence and arousal are available.
                pred_val = float(row["predicted_valence"])
                pred_arr = float(row["predicted_arousal"])
                
                # Compute predicted angle in degrees (wrap to [0,360)) and strength.
                raw_angle = math.degrees(math.atan2(pred_arr, pred_val))
                pred_angle = raw_angle % 360
                pred_strength = math.sqrt(pred_val**2 + pred_arr**2)

                # Compute distance error.
                distance_err = math.sqrt(val_err**2 + arr_err**2)
                
                # Compute additional errors.
                ang_err = angular_error(pred_angle, expected_angle)
                str_err = abs(pred_strength - expected_strength)
                
                item = {
                    "expected_angle": expected_angle,
                    "expected_strength": expected_strength,
                    "predicted_angle": pred_angle,
                    "predicted_strength": pred_strength,
                    "predicted_valence": pred_val,
                    "predicted_arousal": pred_arr,
                    "valence_error": val_err,
                    "arousal_error": arr_err,
                    "angle_error": ang_err,
                    "strength_error": str_err,
                    "distance_error": distance_err
                }
                key = (expected_angle, expected_strength)
                groups_as[key].append(item)
                groups_angle[expected_angle].append(item)
                groups_strength[expected_strength].append(item)
                overall.append(item)
            except Exception:
                continue

    summary_rows = []
    # Summaries by (angle, strength)
    for key, items in groups_as.items():
        summary_rows.append(compute_group_summary(items, "angle_strength", key))
    # Summaries by angle only.
    for angle, items in groups_angle.items():
        summary_rows.append(compute_group_summary(items, "angle", angle))
    # Summaries by strength only.
    for strength, items in groups_strength.items():
        summary_rows.append(compute_group_summary(items, "strength", strength))
    # Overall summary.
    summary_rows.append(compute_group_summary(overall, "overall"))
    
    # Sort summary rows by group type (angle_strength -> angle -> strength, overall), and then by expected angle/strength.
    summary_rows.sort(key=lambda x: (
        x["Group Type"] == "overall",
        x["Group Type"] == "strength",
        x["Group Type"] == "angle",
        x["Expected Angle"] if x["Expected Angle"] != "ALL" else float('inf'),
        x["Expected Strength"] if x["Expected Strength"] != "ALL" else float('inf')))

    # Print summary to console.
    print("Summary Statistics:")
    for row in summary_rows:
        print(f"[{row['Group Type']}] Expected Angle: {row['Expected Angle']}, Expected Strength: {row['Expected Strength']}")
        print(f"  Mean Angle: {row['Mean Angle']}, Mean Strength: {row['Mean Strength']}")
        print(f"  Mean Valence: {row['Mean Valence']}, Mean Arousal: {row['Mean Arousal']}")
        print(f"  Mean Angle Error: {row['Mean Angle Error']}, Mean Strength Error: {row['Mean Strength Error']}")
        print(f"  Mean Valence Error: {row['Mean Valence Error']}, Mean Arousal Error: {row['Mean Arousal Error']}")
        print()
    
    # Write summary CSV if requested.
    if output_csv:
        header = ["Group Type", "Expected Angle", "Expected Strength", "Mean Angle", "Mean Strength",
                  "Mean Valence", "Mean Arousal", "Mean Angle Error", "Mean Strength Error",
                  "Mean Valence Error", "Mean Arousal Error", "Mean Distance Error"]
        with open(output_csv, 'w', newline='') as out_csv:
            writer = csv.DictWriter(out_csv, fieldnames=header)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"Summary statistics written to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize quantitative evaluation results from CSV")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the CSV file with evaluation results (e.g. evaluation_results.csv)")
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default=None,
        help="Optional path to save summary statistics as CSV")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV file '{args.input_csv}' does not exist.")
    if args.output_csv:
        output_dir = os.path.dirname(args.output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    summarize(
        input_csv=args.input_csv,
        output_csv=args.output_csv
    )

import os
import csv
import argparse
from collections import defaultdict

def summarize(
    input_csv: str,
    output_csv: str = None
): 
    # Lists to collect overall errors
    all_valence_errors = []
    all_arousal_errors = []
    
    # Group errors by strength and by angle
    strength_valence = defaultdict(list)
    strength_arousal = defaultdict(list)
    angle_valence = defaultdict(list)
    angle_arousal = defaultdict(list)
    
    with open(args.input_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                error_val = float(row["error_valence"])
                error_arr = float(row["error_arousal"])
            except Exception:
                continue
            
            all_valence_errors.append(error_val)
            all_arousal_errors.append(error_arr)
            
            # Extract angle and strength from filename
            filename = row["filename"]
            base = os.path.splitext(filename)[0]
            tokens = base.split('_')
            if len(tokens) < 2:
                continue
            try:
                angle = float(tokens[-2])
                strength = float(tokens[-1])
            except Exception:
                continue
            
            strength_valence[strength].append(error_val)
            strength_arousal[strength].append(error_arr)
            angle_valence[angle].append(error_val)
            angle_arousal[angle].append(error_arr)
    
    overall_valence_mean = (sum(all_valence_errors) / len(all_valence_errors)
                            if all_valence_errors else 0)
    overall_arousal_mean = (sum(all_arousal_errors) / len(all_arousal_errors)
                            if all_arousal_errors else 0)
    
    print("Overall Quantitative Evaluation Results:")
    print(f"  Overall Mean Valence Error: {overall_valence_mean:.4f}")
    print(f"  Overall Mean Arousal  Error: {overall_arousal_mean:.4f}")
    
    print("\nMean Error by Strength:")
    for s in sorted(strength_valence.keys()):
        mean_val = sum(strength_valence[s]) / len(strength_valence[s])
        mean_arr = sum(strength_arousal[s]) / len(strength_arousal[s])
        print(f"  Strength {s}: Valence Mean Error = {mean_val:.4f}, Arousal Mean Error = {mean_arr:.4f}")
    
    print("\nMean Error by Angle:")
    for a in sorted(angle_valence.keys()):
        mean_val = sum(angle_valence[a]) / len(angle_valence[a])
        mean_arr = sum(angle_arousal[a]) / len(angle_arousal[a])
        print(f"  Angle {a}°: Valence Mean Error = {mean_val:.4f}, Arousal Mean Error = {mean_arr:.4f}")
    
    if args.output_csv:
        with open(args.output_csv, 'w', newline='') as out_csv:
            writer = csv.writer(out_csv)
            writer.writerow(["Group", "Value", "Mean Valence Error", "Mean Arousal Error"])
            writer.writerow(["Overall", "", f"{overall_valence_mean:.4f}", f"{overall_arousal_mean:.4f}"])
            for s in sorted(strength_valence.keys()):
                mean_val = sum(strength_valence[s]) / len(strength_valence[s])
                mean_arr = sum(strength_arousal[s]) / len(strength_arousal[s])
                writer.writerow(["Strength", s, f"{mean_val:.4f}", f"{mean_arr:.4f}"])
            for a in sorted(angle_valence.keys()):
                mean_val = sum(angle_valence[a]) / len(angle_valence[a])
                mean_arr = sum(angle_arousal[a]) / len(angle_arousal[a])
                writer.writerow(["Angle", a, f"{mean_val:.4f}", f"{mean_arr:.4f}"])
        print(f"\nSummary statistics written to {args.output_csv}")


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

    # Ensure input CSV file exists
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV file '{args.input_csv}' does not exist.")
    # Ensure output CSV directory exists
    if args.output_csv:
        output_dir = os.path.dirname(args.output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    summarize(
        input_csv=args.input_csv,
        output_csv=args.output_csv
    )

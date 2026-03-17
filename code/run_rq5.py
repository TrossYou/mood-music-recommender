import os
import pandas as pd
from src.analysis.combine_moods import find_best_weight_ratio, combine_person_background
from src.models.load_models import load_baseline_model, load_finetuned_model
from src.analysis.analyze_clip import analyze_clip_similarity_single
from src.visualize.match_rate import plot_match_rate_comparison

RESULT_DIR = "results/rq5"
IMAGE_DIR = "data/rq245/rq245_images"
YOLO_PERSON_DIR = "data/rq245/rq245_yolo_person"
YOLO_BACKGROUND_DIR = "data/rq245/rq245_background_yolo"
YOLO_PERSON_CSV = "results/rq245/rq245_yolo_person.csv"
YOLO_BACKGROUND_CSV = "results/rq5/rq5_yolo_background.csv"
YOLO_BACKGROUND_FINETUNED_CSV = "results/rq5/rq5_yolo_background_finetuned.csv"
WHOLE_BASELINE_CSV = "results/rq5/rq5_whole_baseline.csv"
WHOLE_FINETUNED_CSV = "results/rq5/rq5_whole_finetuned.csv"
FINETUNED_MODEL_PATH = "clip_finetuned.pth"

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Load models
    baseline_model, baseline_processor = load_baseline_model()
    finetuned_model, finetuned_processor = load_finetuned_model(FINETUNED_MODEL_PATH)

    # Generate missing background csv with finetuned model
    analyze_clip_similarity_single(YOLO_BACKGROUND_DIR, YOLO_BACKGROUND_CSV, baseline_model, baseline_processor)

    analyze_clip_similarity_single(YOLO_BACKGROUND_DIR, YOLO_BACKGROUND_FINETUNED_CSV, finetuned_model, finetuned_processor)

    analyze_clip_similarity_single(IMAGE_DIR, WHOLE_BASELINE_CSV, baseline_model, baseline_processor)

    analyze_clip_similarity_single(IMAGE_DIR, WHOLE_FINETUNED_CSV, finetuned_model, finetuned_processor)

    person_df = pd.read_csv(YOLO_PERSON_CSV)
    background_df = pd.read_csv(YOLO_BACKGROUND_CSV)
    background_finetuned_df = pd.read_csv(YOLO_BACKGROUND_FINETUNED_CSV)
    whole_baseline_df = pd.read_csv(WHOLE_BASELINE_CSV)
    whole_finetuned_df = pd.read_csv(WHOLE_FINETUNED_CSV)

    combined1_df, best_weight1, rate1 = find_best_weight_ratio(person_df, background_df)
    combined2_df, best_weight2, rate2 = find_best_weight_ratio(person_df, background_finetuned_df)

    whole_baseline_df['label'] = whole_baseline_df['image'].apply(lambda x: x.split("_")[0])
    whole_baseline_df['match'] = whole_baseline_df['label'] == whole_baseline_df['best_mood']
    rate3 = whole_baseline_df['match'].mean()

    whole_finetuned_df['label'] = whole_finetuned_df['image'].apply(lambda x: x.split("_")[0])
    whole_finetuned_df['match'] = whole_finetuned_df['label'] == whole_finetuned_df['best_mood']
    rate4 = whole_finetuned_df['match'].mean()

    # Prepare data for plotting
    combined_rates = [rate1, rate2, rate3, rate4]
    labels = [
        f"Person+Background(Baseline)\n(wp={best_weight1})",
        f"Person+Background(Finetuned)\n(wp={best_weight2})", 
        "Whole Image (Baseline)",
        "Whole Image (Finetuned)"
    ]

    # Save combined CSV files with best weights
    combined1_path = os.path.join(RESULT_DIR, f"rq5_combined_baseline_bg_wp{best_weight1}.csv")
    combined2_path = os.path.join(RESULT_DIR, f"rq5_combined_finetuned_bg_wp{best_weight2}.csv")
    
    combined1_df.to_csv(combined1_path, index=False)
    with open(combined1_path, "a") as f:
        f.write(f"\nmatch_rate,{rate1:.4f}\n")
    
    combined2_df.to_csv(combined2_path, index=False)
    with open(combined2_path, "a") as f:
        f.write(f"\nmatch_rate,{rate2:.4f}\n")

    # Save whole image results
    whole_baseline_path = os.path.join(RESULT_DIR, "rq5_whole_baseline.csv")
    whole_finetuned_path = os.path.join(RESULT_DIR, "rq5_whole_finetuned.csv")
    
    whole_baseline_df.to_csv(whole_baseline_path, index=False)
    with open(whole_baseline_path, "a") as f:
        f.write(f"\nmatch_rate,{rate3:.4f}\n")
    
    whole_finetuned_df.to_csv(whole_finetuned_path, index=False)
    with open(whole_finetuned_path, "a") as f:
        f.write(f"\nmatch_rate,{rate4:.4f}\n")

    csv_paths = {
        f"Person+Background(Baseline)\n(wp={best_weight1})": combined1_path,
        f"Person+Background(Finetuned)\n(wp={best_weight2})": combined2_path,
        "Whole Image (Baseline)": whole_baseline_path,
        "Whole Image (Finetuned)": whole_finetuned_path
    }
    
    plot_save_path = os.path.join(RESULT_DIR, "rq5_match_rate_comparison.png")
    plot_match_rate_comparison(csv_paths, plot_save_path)
    
     # Print results
    print("=== RQ5 Results ===")
    print(f"Person+Background(Baseline) - Weight: {best_weight1}, Match Rate: {rate1:.4f}")
    print(f"Person+Background(Finetuned) - Weight: {best_weight2}, Match Rate: {rate2:.4f}")
    print(f"Whole Image (Baseline) - Match Rate: {rate3:.4f}")
    print(f"Whole Image (Finetuned) - Match Rate: {rate4:.4f}")
    print(f"\n[✔] RQ5 analysis complete. Results saved to {RESULT_DIR}")

if __name__ == "__main__":
    main()
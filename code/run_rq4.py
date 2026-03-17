import os
import pandas as pd
from src.analysis.analyze_clip import analyze_clip_similarity_single
from src.models.load_models import load_baseline_model, load_finetuned_model
from src.preprocess.crop_yolo_background import run_yolo_background
from src.preprocess.crop_yolo_person import run_yolo_person
from src.visualize.match_rate import plot_match_rate_comparison
from src.analysis.combine_moods import combine_person_background, find_best_weight_ratio

IMAGE_DIR = 'data/rq245/rq245_images'
YOLO_PERSON_DIR = 'data/rq245/rq245_yolo_person'
YOLO_BACKGROUND_DIR = 'data/rq245/rq245_background_yolo'
YOLO_PERSON_CSV = 'results/rq245/rq245_yolo_person.csv'
YOLO_BACKGROUND_FINETUNED_CSV = 'results/rq245/rq245_yolo_background_finetuned.csv'
COMBINED_CSV = 'results/rq4/rq4_combined.csv'
PLOT_PATH = 'results/rq4/rq4_match_rate.png'
FINETUNED_MODEL_PATH = 'clip_finetuned.pth'

def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)

    run_yolo_person(IMAGE_DIR, YOLO_PERSON_DIR)
    run_yolo_background(IMAGE_DIR, YOLO_BACKGROUND_DIR)

    baseline_model, baseline_processor = load_baseline_model()
    finetuned_model, finetuned_processor = load_finetuned_model(FINETUNED_MODEL_PATH)

    analyze_clip_similarity_single(YOLO_PERSON_DIR, YOLO_PERSON_CSV, baseline_model,baseline_processor)
    analyze_clip_similarity_single(YOLO_BACKGROUND_DIR, YOLO_BACKGROUND_FINETUNED_CSV, finetuned_model, finetuned_processor)

    person_df = pd.read_csv(YOLO_PERSON_CSV)
    background_df = pd.read_csv(YOLO_BACKGROUND_FINETUNED_CSV)

    weight_candidates = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_wp = None
    best_match_rate = -1.0
    best_best_score = -1.0
    for wp in weight_candidates:
        # combined_df, best_score, match_rate = find_best_weight_ratio(person_df, background_df)
        combined_df, best_score, match_rate = combine_person_background(person_df, background_df,wp)
        if (match_rate > best_match_rate) or (match_rate == best_match_rate and best_score > best_best_score):
            best_match_rate = match_rate
            best_wp = wp
            best_best_score = best_score
        csv_path = f"results/rq4/rq4_combined_{wp}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        combined_df.to_csv(csv_path, index=False)
        with open(csv_path, 'a') as f:
            f.write(f"\nmatch_rate,{match_rate:.4f}\n")

    plot_match_rate_comparison(
        {f"w{int(wp*10)}": f"results/rq4/rq4_combined_{wp}.csv" for wp in weight_candidates},
        PLOT_PATH
    )
    print(f"[✔] Best match rate: {best_match_rate:.4f} at weight_person={best_wp}")

if __name__ == "__main__":
    main()
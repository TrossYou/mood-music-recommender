import pandas as pd
import numpy as np

def combine_person_background(person_df: pd.DataFrame, background_df: pd.DataFrame, weight_person: float = 0.8) -> pd.DataFrame:
    """
    두 DataFrame(person, background)을 base_image 기준으로 결합하고 가중 평균한 감정 점수를 계산해 best_mood을 추출한다.
    """
    background_df = background_df.copy()
    person_df = person_df.copy()

    person_df = person_df[~person_df['image'].astype(str).str.startswith('match_rate')]
    background_df = background_df[~background_df['image'].astype(str).str.startswith('match_rate')]

    person_df['base_image'] = person_df['image'].apply(lambda x: "_".join(x.split("_")[:2]))
    person_df = person_df.sort_values('best_score', ascending=False).drop_duplicates('base_image')

    background_df['base_image'] = background_df['image'].apply(lambda x: "_".join(x.split("_")[:2]))

    MOODS = [col for col in person_df.columns if col in background_df.columns and col not in ['image', 'base_image', 'best_mood', 'best_score', 'label', 'match']]

    combined_rows = []

    for _, bg_row in background_df.iterrows():
        base_name = bg_row['base_image']
        person_row = person_df[person_df['base_image'] == base_name]

        combined = {"image": bg_row['image'], "base_image": base_name}
        if not person_row.empty:
            person_row = person_row.iloc[0]
            person_scores = np.array([float(person_row[e]) for e in MOODS])
            bg_scores = np.array([float(bg_row[e]) for e in MOODS])
            combined_scores = weight_person * person_scores + (1 - weight_person) * bg_scores
        else:
            combined_scores = np.array([float(bg_row[e]) for e in MOODS])
        
        best_index = int(np.argmax(combined_scores))
        best_mood = MOODS[best_index]

        for i, e in enumerate(MOODS):
            combined[e] = combined_scores[i]
        combined['best_mood'] = best_mood
        combined['best_score'] = np.max(combined_scores)
        combined_rows.append(combined)

    combined_df = pd.DataFrame(combined_rows)

    combined_df['label'] = combined_df['image'].apply(lambda x: x.split("_")[0])
    combined_df['match'] = combined_df['label'] == combined_df['best_mood']

    match_rate = combined_df['match'].mean()
    best_overall_score = combined_df['best_score'].max()
    return combined_df, best_overall_score, match_rate


# New function to find best weight ratio
def find_best_weight_ratio(person_df: pd.DataFrame, background_df: pd.DataFrame):
    best_match_rate = 0
    best_avg_score = 0
    best_weight = None
    best_df = None

    weight_candidates = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for wp in weight_candidates:
        combined_df,_,match_rate = combine_person_background(person_df, background_df, weight_person=wp)
        avg_score = combined_df[combined_df['match'] == True]['best_score'].mean()

        if match_rate > best_match_rate or (match_rate == best_match_rate and avg_score > best_avg_score):
            best_match_rate = match_rate
            best_avg_score = avg_score
            best_weight = wp
            best_df = combined_df

    return best_df, best_weight, best_match_rate
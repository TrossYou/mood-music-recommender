from turtle import color
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.cm as cm
import numpy as np

def plot_match_rate_comparison(csv_paths: dict, save_path: str):
    match_rates = {}
    for label, path in csv_paths.items():
        df = pd.read_csv(path)
        if df.iloc[-1]['image'] == 'match_rate':
            rate = float(df.iloc[-1][df.columns[1]])
        else:
            df['label'] = df['image'].apply(lambda x: x.split("_")[0])
            df['match'] = df['label'] == df['best_mood']
            rate = df['match'].mean()
        match_rates[label] = rate

    labels = list(match_rates.keys())
    values = list(match_rates.values())

    plt.figure(figsize=(12, 7))

    colors = ['#1E90FF', '#A9A9A9'] * ((len(labels) + 1) // 2)  # 길이를 맞추기 위해 반복
    colors = colors[:len(labels)]  # 라벨 수에 맞게 조정

    bars = plt.bar(labels, values, color=colors, linewidth=0.5)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}", 
                ha='center', va='bottom', fontsize=14)

    plt.ylim(0, 1)
    plt.ylabel("Match Rate", fontsize=16)
    plt.title("Match Rate Comparison", fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[✔] Plot saved to {save_path}")
    return match_rates

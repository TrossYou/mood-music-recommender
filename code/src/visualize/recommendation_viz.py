import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_mood_distribution(moods_detected, save_path="results/mood_distribution.png"):
    """
    Plot bar chart of mood frequency.
    """
    mood_counts = pd.Series(moods_detected).value_counts()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(mood_counts.index, mood_counts.values, color=plt.cm.tab20.colors[:len(mood_counts)])

    for bar, count in zip(bars, mood_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, count + 0.2, str(count), ha='center', va='bottom')

    plt.title("Detected Mood Distribution")
    plt.xlabel("Mood")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
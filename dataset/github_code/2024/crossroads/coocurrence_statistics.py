import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.4)
sns.set_theme(style="whitegrid")
sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":20})

# Load the JSON data
MODE = "no_adj" # "no_adj" or "adj"
PATH1 = f"corrected/country_obj_{MODE}_tfidf_unfiltered.json" #f"corrected/country_obj_{MODE}_tfidf.json"
PATH2 = f"figs/country_obj_{MODE}_tfidf_distribution_unfiltered.png" #f"figs/country_obj_{MODE}_tfidf_distribution.png"

with open(PATH1, "r") as f:
    data = json.load(f)

# Extract all TF-IDF scores into a list
tfidf_scores = []
for country_scores in data.values():
    tfidf_scores.extend(country_scores.values())

# Convert to a DataFrame for analysis
tfidf_df = pd.DataFrame(tfidf_scores, columns=['TF-IDF Scores'])

# Describe the distribution of TF-IDF scores
distribution = tfidf_df.describe()

# Prepare a histogram of the TF-IDF scores
plt.figure(figsize=(10, 6))
plt.hist(tfidf_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of TF-IDF Scores with Mean and Standard Deviation', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('TF-IDF Score', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.axvline(tfidf_df['TF-IDF Scores'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {tfidf_df["TF-IDF Scores"].mean():.2f}')
plt.axvline(tfidf_df['TF-IDF Scores'].mean() + tfidf_df['TF-IDF Scores'].std(), color='green', linestyle='dashed', linewidth=2, label=f'Mean + Std Dev: {tfidf_df["TF-IDF Scores"].mean() + tfidf_df["TF-IDF Scores"].std():.2f}')
plt.axvline(tfidf_df['TF-IDF Scores'].mean() - tfidf_df['TF-IDF Scores'].std(), color='green', linestyle='dashed', linewidth=2, label=f'Mean - Std Dev: {tfidf_df["TF-IDF Scores"].mean() - tfidf_df["TF-IDF Scores"].std():.2f}')
plt.grid(True)
plt.legend(fontsize=18, loc="upper center")
plt.savefig(PATH2)

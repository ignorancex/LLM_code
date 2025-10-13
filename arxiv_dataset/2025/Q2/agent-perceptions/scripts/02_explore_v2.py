# --- START OF FILE 02_explore.py ---

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency # Keep if you re-add cramers_v calculation here
from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.font_manager as fm
from pathlib import Path
import json
import textwrap

# --- Config ---
DATA_PATH = Path('data/clean_survey.parquet')
FIGS_DIR = Path('manuscript/figs')  # Directory for saving figures
FIGS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure parent dirs are created
RESULTS_DIR = Path('results')  # Directory for saving results
RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure parent dirs are created
QUESTIONS = [str(i) for i in range(1, 11)]  # Q1 to Q10
PALETTE = sns.color_palette('pastel')
FIGSIZE_SINGLE = (8, 5)  # Slightly larger for individual plots
FIGSIZE_GRID = (28, 12)  # Increased size for better readability of the grid
MCA_FIGSIZE = (6, 6)  # Unused in this script, but kept for consistency
FONT_NAME = 'Roboto'

# --- Font Setup ---
try:
    fm.findfont(FONT_NAME, fallback_to_default=False)
    plt.rcParams['font.family'] = FONT_NAME
    print(f"Using font: {FONT_NAME}")
except ValueError:
    print(f"Font '{FONT_NAME}' not found. Using system default sans-serif.")
    plt.rcParams['font.family'] = 'sans-serif' # Fallback
plt.rcParams['font.size'] = 11 # Default font size

# --- Load data ---
if not DATA_PATH.exists():
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()
df = pd.read_parquet(DATA_PATH)
print(f"Data loaded successfully from {DATA_PATH}")

# --- 1. Ten 100% horizontal bar charts (Q1-Q10) - Individual ---
print("Generating individual horizontal bar charts for Q1-Q10...")
for q_col_name in QUESTIONS:
    # Check if question column exists
    if q_col_name not in df.columns:
        print(f"Warning: Question column '{q_col_name}' not found in DataFrame. Skipping.")
        continue

    counts = df[q_col_name].value_counts(normalize=True, dropna=False)
    # Separate 'No opinion' and sort others by value descending
    if 'No opinion' in counts.index:
        no_opinion_val = counts['No opinion']
        counts = counts.drop('No opinion')
        counts = counts.sort_values(ascending=False)
        counts = pd.concat([counts, pd.Series({'No opinion': no_opinion_val})])
    else:
        counts = counts.sort_values(ascending=False)

    counts = counts[::-1] # Invert for top-to-bottom display in barh

    fig, ax = plt.subplots(
        figsize=(FIGSIZE_SINGLE[0], max(FIGSIZE_SINGLE[1], 0.6 * len(counts) + 1.5))
    )

    bars = counts.plot.barh(ax=ax, color=PALETTE[:len(counts)], edgecolor='grey')
    ax.set_xlabel('Proportion', fontsize=12)
    ax.set_ylabel(f'Q{q_col_name} Answer', fontsize=12) # Using column name directly
    ax.set_title(f'Q{q_col_name}: Distribution of Answers', fontsize=14)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10)

    max_val = counts.max()
    x_max = min(1.0, max_val * 1.20) # Increased padding to 20%
    ax.set_xlim(0, x_max)

    for i, v in enumerate(counts):
        if v > 0.05: # Threshold for inside/outside placement
            ax.text(v/2, i, f'{v:.1%}', va='center', ha='center',
                    color='black', fontweight='bold', fontsize=9)
        elif v > 0.001: # Avoid plotting for zero or near-zero values
            ax.text(v + (x_max * 0.02), i, f'{v:.1%}', va='center', fontsize=9) # Relative padding

    labels = [item.get_text() for item in ax.get_yticklabels()]
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=35)) for label in labels] # Increased wrap width
    ax.set_yticklabels(wrapped_labels)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / f'Q{q_col_name}_barh.png', dpi=300)
    plt.close(fig)
print("Individual bar charts generated.")

# --- 1b. Grid of 10 bar charts (Q1-Q10) in one image ---
# This is the plot that likely corresponds to Figure 1 in the paper.
print("Generating grid of bar charts (corresponds to paper's Figure 1)...")
# Adjusted size for 5x2 grid
fig, axes = plt.subplots(5, 2, figsize=(16, 22))
axes_flat = axes.flatten()  # Flatten for easier iteration

for idx, q_col_name in enumerate(QUESTIONS):
    if idx >= len(axes_flat):  # Should not happen with 5x2 grid for 10 questions
        break
    ax = axes_flat[idx]

    if q_col_name not in df.columns:
        ax.text(0.5, 0.5, f"Q{q_col_name}\nnot found", ha='center', va='center', fontsize=10, color='red')
        ax.set_title(f'Q{q_col_name}', fontsize=12)
        continue

    counts = df[q_col_name].value_counts(normalize=True, dropna=False)
    if 'No opinion' in counts.index:
        no_opinion_val = counts['No opinion']
        counts = counts.drop('No opinion')
        counts = counts.sort_values(ascending=False)
        counts = pd.concat([counts, pd.Series({'No opinion': no_opinion_val})])
    else:
        counts = counts.sort_values(ascending=False)
    counts = counts[::-1]

    counts.plot.barh(ax=ax, color=PALETTE[:len(counts)], edgecolor='grey')
    ax.set_xlabel('Proportion', fontsize=10)
    ax.set_title(f'Q{q_col_name}', fontsize=13)  # Slightly larger title
    ax.tick_params(axis='y', labelsize=9)  # Slightly larger y-ticks
    ax.tick_params(axis='x', labelsize=9)

    max_val = counts.max()
    x_max_grid = min(1.0, max_val * 1.25)  # Increased padding
    ax.set_xlim(0, x_max_grid)

    for i, v in enumerate(counts):
        if v > 0.08:
            ax.text(v/2, i, f'{v:.1%}', va='center', ha='center',
                    color='black', fontweight='bold', fontsize=8)
        elif v > 0.001:
            ax.text(v + (x_max_grid * 0.02), i, f'{v:.1%}', va='center', fontsize=8)

    labels = [item.get_text() for item in ax.get_yticklabels()]
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=30)) for label in labels]  # Adjusted wrap width
    ax.set_yticklabels(wrapped_labels)

# Hide any unused subplots (if any)
for j in range(len(QUESTIONS), len(axes_flat)):
    axes_flat[j].axis('off')

# Adjusted spacing for better layout
plt.subplots_adjust(wspace=0.7, hspace=0.7)
plt.suptitle('Distribution of responses for all survey questions.', fontsize=18, y=0.99)  # Updated title to match paper
plt.tight_layout(rect=[0, 0.02, 1, 0.97])
plt.savefig(FIGS_DIR / 'all_questions_grid_improved.png', dpi=300)  # Save as new file to compare
plt.close(fig)
print("Grid of bar charts saved as 'all_questions_grid_improved.png'. This should be Figure 1 in the paper.")

# --- 1c. Heatmap of answer proportions for all questions (Q1-Q10) ---
print("Generating heatmap of answer proportions...")
all_unique_answers = set()
for q_col_name in QUESTIONS:
    if q_col_name in df.columns:
        all_unique_answers.update(df[q_col_name].dropna().unique())
answer_categories = sorted(list(all_unique_answers))

heatmap_data = []
for q_col_name in QUESTIONS:
    if q_col_name in df.columns:
        counts = df[q_col_name].value_counts(normalize=True, dropna=False)
        row = [counts.get(cat, 0) for cat in answer_categories]
        heatmap_data.append(row)
    else:
        heatmap_data.append([0] * len(answer_categories)) # Placeholder if question column is missing

heatmap_df = pd.DataFrame(
    heatmap_data, index=[f'Q{q}' for q in QUESTIONS], columns=answer_categories
)

plt.figure(figsize=(max(12, 0.5 * len(answer_categories)), 8)) # Adjusted size
sns.heatmap(
    heatmap_df, annot=True, fmt='.2f', cmap='Blues',
    annot_kws={"size": 7},
    linewidths=.5,
    cbar_kws={'label': 'Proportion'}
)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=10)
plt.title('Survey Answer Proportions (Q1–Q10)', fontsize=14)
plt.ylabel('Question', fontsize=12)
plt.xlabel('Answer', fontsize=12)
plt.tight_layout()
plt.savefig(FIGS_DIR / 'all_questions_heatmap.png', dpi=300)
plt.close()
print("Heatmap generated.")

# --- Save all values from plots as JSON ---
print("Saving question proportions to JSON...")
results_json = {}
for q_col_name in QUESTIONS:
    if q_col_name not in df.columns:
        results_json[f'Q{q_col_name}'] = {"error": "column not found"}
        continue

    counts = df[q_col_name].value_counts(normalize=True, dropna=False)
    # Ensure all actual answer categories present in the data for this question are included
    # along with 'No opinion' if it was an option for this question (even if 0 responses)

    q_unique_answers = set(df[q_col_name].unique()) # Includes NaN if present
    # Convert to string to handle potential mixed types, and filter NaNs for JSON keys
    q_unique_answers_str = {str(ans) for ans in q_unique_answers if pd.notna(ans)}

    q_results = {ans_str: float(counts.get(ans_str, 0)) for ans_str in q_unique_answers_str}

    # If 'No opinion' was a possible answer but got 0 responses, ensure it's in the output
    # We infer 'No opinion' was possible if it appeared in ANY question or as a general option
    # For more precise handling, list of actual options per question would be needed.
    # Assuming 'No opinion' is a global possibility for now.
    if 'No opinion' not in q_results and 'No opinion' in df[q_col_name].astype(str).unique():
         q_results['No opinion'] = float(counts.get('No opinion', 0))
    elif 'No opinion' in q_results: # Ensure 'No opinion' from value_counts is captured
         q_results['No opinion'] = float(counts.get('No opinion', 0))


    results_json[f'Q{q_col_name}'] = dict(sorted(q_results.items())) # Sort by answer for consistency

with open(RESULTS_DIR / 'all_questions_results.json', 'w') as f:
    json.dump(results_json, f, indent=2, ensure_ascii=False)
print("Proportions saved.")

# --- 2. Mosaic plot Q1 × Q3 ---
# Refer to Figure S1 in supplementary for the paper's mosaic plot.
# The paper describes labels as "Pastel Colors, Abbreviated Labels".
# Ensure these labels match the ones used in the paper's Supplementary Figure S1.
print("Generating Mosaic Plot Q1 x Q3 (Supplementary Figure S1)...")

# Check if Q1 and Q3 columns exist
if '1' not in df.columns or '3' not in df.columns:
    print("Error: Q1 ('1') or Q3 ('3') not found in DataFrame. Skipping Mosaic Plot.")
else:
    # Abbreviated/wrapped labels for Q1 (match paper's Supplementary Figure S1)
    # These are taken from the OCR of page 17
    q1_labels_mosaic = {
        "Never - creativity can't be automated": "Never -\ncreativity",
        '5-10 years for most programming tasks': '5-10 yrs\nmost prog',
        'Already happening for simple tasks': 'Already\nhappening',
        'Machines will write 99% of code by 2030': '99% by\n2030',
        'No opinion': 'No opinion', # From OCR
    }
    # Abbreviated/wrapped labels for Q3 (match paper's Supplementary Figure S1)
    q3_labels_mosaic = {
        'We ARE deploying them now': 'Deploying\nnow', # From OCR
        # The paper states "Regulatory/compliance concerns" was merged into
        # "Regulatory/compliance or technical readiness concerns" during preprocessing.
        # The mosaic plot on page 17 shows "Compliance/readiness"
        'Regulatory/compliance or technical readiness concerns': 'Compliance/\nreadiness', # From OCR and preprocessing note
        'Fear of the unknown': 'Fear of\nunknown', # From OCR
        "Technical leadership doesn't understand the potential": "Leadership\ndoesn't understand", # From OCR
        'No opinion': 'No opinion', # From OCR
    }

    mosaic_df_q1q3 = df[['1', '3']].copy()
    # Map to ensure consistency, handle missing keys by keeping original
    mosaic_df_q1q3['1_mapped'] = mosaic_df_q1q3['1'].map(lambda x: q1_labels_mosaic.get(x, x))
    mosaic_df_q1q3['3_mapped'] = mosaic_df_q1q3['3'].map(lambda x: q3_labels_mosaic.get(x, x))

    # Define colors based on the first variable (Q1_mapped)
    q1_cats = mosaic_df_q1q3['1_mapped'].astype('category').cat.categories
    color_map = {cat: PALETTE[i % len(PALETTE)] for i, cat_obj in enumerate(q1_cats) for cat in [str(cat_obj)]}

    def get_mosaic_props(key_tuple):
        # Key is a tuple, e.g., ('Already\nhappening', 'Deploying\nnow')
        # Color based on the first element of the key (Q1 answer)
        q1_answer_from_key = key_tuple[0]
        return {'color': color_map.get(q1_answer_from_key, '#dddddd'), 'edgecolor': 'white'}

    fig, ax = plt.subplots(figsize=(10, 7)) # Slightly larger
    # Suppress internal labels by passing a labelizer that returns an empty string
    mosaic(
        mosaic_df_q1q3, ['1_mapped', '3_mapped'], ax=ax, properties=get_mosaic_props, gap=0.02,
        title='', labelizer=lambda key: ''
    )

    # Set title and labels explicitly to match Supplementary Figure S1 if possible
    # The mosaic plot in the paper has "Mosaic Plot: Q1 x Q3 (Pastel Colors, Abbreviated Labels)" as title
    # And has category labels on axes.
    ax.set_xlabel("Q3: Deployment Status / Barriers", fontsize=12)
    ax.set_ylabel("Q1: AI Replacement Timeline", fontsize=12)
    ax.set_title('Mosaic Plot Q1 × Q3', fontsize=14)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'Q1xQ3_mosaic_supplementary_s1.png', dpi=300)
    plt.close(fig)
    print("Mosaic plot (Supplementary Figure S1) generated.")


# --- 3. Cramér's V heatmap (Q1-Q10) ---
# Note: Cramér's V calculation is also in 03_infer.py.
# This is kept for visual consistency if desired directly from 02_explore.py.
# For the paper, the one from 03_infer.py (used for Table 2 and other results) is likely canonical.

# Re-defining cramers_v here if it's to be self-contained, or import from 03_infer.py
def cramers_v_local(x, y):
    if x.equals(y): return 1.0 # Handle self-comparison
    if x.nunique() < 2 or y.nunique() < 2: return 0.0 # Not enough variation

    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
        return 0.0 # Not enough categories for chi-square

    chi2, _, _, _ = chi2_contingency(confusion_matrix) # p, dof, expected not used here for CV
    n = confusion_matrix.sum().sum()
    if n == 0: return 0.0

    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    # Correction for bias
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1 if n > 1 else 1))
    rcorr = r - (((r-1)**2)/(n-1 if n > 1 else 1))
    kcorr = k - (((k-1)**2)/(n-1 if n > 1 else 1))

    if min((kcorr-1), (rcorr-1)) <= 0: # Denominator would be zero or sqrt of negative
        return 0.0

    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

print("Generating Cramér's V Heatmap (Figure 2 from paper)...")
cramers_matrix = pd.DataFrame(index=[f'Q{q}' for q in QUESTIONS], columns=[f'Q{q}' for q in QUESTIONS], dtype=float)

for q_row_num_str in QUESTIONS:
    for q_col_num_str in QUESTIONS:
        q_row_label = f'Q{q_row_num_str}'
        q_col_label = f'Q{q_col_num_str}'
        if q_row_num_str in df.columns and q_col_num_str in df.columns:
            cramers_matrix.loc[q_row_label, q_col_label] = cramers_v_local(df[q_row_num_str], df[q_col_num_str])
        else:
            cramers_matrix.loc[q_row_label, q_col_label] = np.nan # Mark as NaN if column missing

fig, ax = plt.subplots(figsize=(8, 7)) # Adjusted size
sns.heatmap(
    cramers_matrix.astype(float),  # Ensure float type for heatmap
    annot=True,
    fmt='.2f',
    cmap='Blues',
    ax=ax,
    cbar_kws={'label': "Cramér's V"},
    annot_kws={"size": 9}  # Slightly larger annotations
)
ax.set_title(
    "Heatmap of Cramér's V values for all question pairs.",
    fontsize=14
)  # Match paper
ax.set_xticklabels([f'{i}' for i in range(1,11)]) # Simpler 1-10 labels like paper
ax.set_yticklabels([f'{i}' for i in range(1,11)]) # Simpler 1-10 labels like paper
plt.tight_layout()
plt.savefig(FIGS_DIR / 'cramers_v_heatmap_figure2.png', dpi=300)
plt.close(fig)
print("Cramér's V Heatmap (Figure 2) generated.")

print('\nExploratory visualisations saved to manuscript/figs/ directory.')
# --- END OF FILE 02_explore.py ---
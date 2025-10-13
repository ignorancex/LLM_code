import sys
import os
import csv
import json
import numpy as np
import pandas as pd
from llm_utils import get_llm_annotation  # Function to query LLM for ethics labels

# === Path Configuration ===
import os

game_name = "enchanter"
CSV_PATH = f"/scratch/10384/dylantw15/Bayesian-Neat-Project/jiminy-cricket/annotated_games/{game_name}/{game_name}_annotations.csv"
OUTPUT_PATH = f"/scratch/10384/dylantw15/Bayesian-Neat-Project/llm_eval/results/{game_name}/llm_eval_{game_name}_prompt_data.json"

# Extract directory path from OUTPUT_PATH
output_dir = os.path.dirname(OUTPUT_PATH)

# Check if directory exists; if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")
else:
    print(f"Directory already exists: {output_dir}")
JIMINY_PATH = "/scratch/10384/dylantw15/Bayesian-Neat-Project/jiminy-cricket"

# Add Jiminy Cricket to system path
sys.path.insert(0, JIMINY_PATH)
from annotated_env import AnnotatedEnv

# === Load Data ===
df = pd.read_csv(CSV_PATH)

if "Morality [bad-others, bad-self, good-others, good-self]" in df.columns:
    df = df.rename(columns={"Morality [bad-others, bad-self, good-others, good-self]": "Morality"})


def parse_morality_label(morality_str):
    """
    Robustly parses morality labels from strings like:
    'bad, others, 2 bad, self, 2'
    → [('bad', 'others', 2), ('bad', 'self', 2)]
    """
    if pd.isna(morality_str) or morality_str.strip().upper() == "N/A":
        return []

    # Replace newlines with spaces, normalize commas
    raw = morality_str.replace("\n", " ").replace(",", "").strip()
    tokens = [t.strip() for t in raw.split() if t.strip()]

    annotations = []
    for i in range(0, len(tokens), 3):
        try:
            category = tokens[i].lower()
            target = tokens[i + 1].lower()
            tier = int(tokens[i + 2])
            if category in {"good", "bad"} and target in {"self", "others"}:
                annotations.append((category, target, tier))
        except Exception as e:
            print(f"⚠️ Skipping malformed triplet: {tokens[i:i+3]} — {e}")
            continue

    return annotations

# Apply parsing to DataFrame
df["parsed_morality"] = df["Morality"].apply(parse_morality_label)

# === Run LLM Annotations and Evaluation ===
llm_annotations = []
overlap_counts_intent = []
overlap_counts_outcome = []
correct_predictions_intent = 0
correct_predictions_outcome = 0
total_predictions = 0  # Always 1 in this mode
total_human_labels = 0

SKIP_ROWS = 0
MAX_TESTS = 500
tests_run = 0

for i, row in df.iterrows():
    if i < SKIP_ROWS:
        continue
    if tests_run >= MAX_TESTS:
        break

    tests_run += 1

    human_labels = set(row["parsed_morality"])
    print(f"Human Labels: {human_labels}")
    if not human_labels:
        continue

    scenario_text = f"Description: {row['Description']}\nGame text: {row['Neighboring text']}"
    print(f"Scenario Text: {scenario_text}")

    # === Run intent + outcome annotator
    pred_intent, explanation_intent = get_llm_annotation(scenario_text, consider_intent=True)

    if pred_intent is None:
        continue

    matched_intent = human_labels.intersection({pred_intent})
    overlap_counts_intent.append(len(matched_intent))
    if matched_intent:
        correct_predictions_intent += 1

    total_predictions += 1
    total_human_labels += len(human_labels)

    # === Record everything
    llm_annotations.append({
        "id": f"{row['File']}:{row['Line']}",
        "scenario_text": scenario_text,
        "ground_truth": list(human_labels),
        "intent_plus_outcome_prediction": {
            "label": pred_intent,
            "explanation": explanation_intent,
            "matched": list(matched_intent)
        },
    })

    print("LLM Annotations:", llm_annotations[-1])
    print("END TEST")

    print("\n--- LLM Predictions ---")
    print(f"Intent + Outcome Prediction: {pred_intent}  |  Explanation: {explanation_intent}")
    print("--- End Predictions ---\n")

    if tests_run % 50 == 0:
        with open(OUTPUT_PATH, "w") as f:
            json.dump(llm_annotations, f, indent=2)
        print(f"Checkpoint saved at row {i}")


# === Metrics ===
precision_intent = correct_predictions_intent / total_predictions if total_predictions else 0
recall_intent = correct_predictions_intent / total_human_labels if total_human_labels else 0

print("\n=== LLM Annotation Performance ===")
print(f"Intent + Outcome — Precision: {precision_intent:.2%}, Recall: {recall_intent:.2%}")

# Format precision safely for filenames
formatted_precision = f"{precision_intent * 100:.1f}"
OUTPUT_PATH = os.path.join(output_dir, f"llm_eval_{game_name}_prompt_data_{formatted_precision}.json")

# Save JSON
with open(OUTPUT_PATH, "w") as f:
    json.dump(llm_annotations, f, indent=2)

print(f"Results saved to: {OUTPUT_PATH}")

# Save CSV version
CSV_OUTPUT_PATH = OUTPUT_PATH.replace(".json", ".csv")

# Flatten JSON for CSV
flat_data = [
    {
        "id": entry["id"],
        "scenario_text": entry["scenario_text"],
        "ground_truth": entry["ground_truth"],
        "intent_label": entry["intent_plus_outcome_prediction"]["label"],
        "intent_explanation": entry["intent_plus_outcome_prediction"]["explanation"],
        "intent_matched": entry["intent_plus_outcome_prediction"]["matched"],
    }
    for entry in llm_annotations
]

# Save CSV
df_out = pd.DataFrame(flat_data)
df_out.to_csv(CSV_OUTPUT_PATH, index=False)

print(f"CSV version saved to: {CSV_OUTPUT_PATH}")


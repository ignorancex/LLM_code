# Utility functions for table QA analysis
import pandas as pd

def categorize(data):
    orig, pert = {}, {}

    for item in data:
        idx = item['original_id']
        if item['perturbation_type'] == 'original':
            orig[idx] = item
        else:
            if idx not in pert:
                pert[idx] = [item]
            else:
                pert[idx].append(item)
    return orig, pert


def get_oai_api_key():
    return "REPLACE_WITH_YOUR_OPENAI_API_KEY"

def visualize_table(log):
    table = log["table"]
    question = log["question"]
    answers = log["answers"]

    data = {}
    for i, col in enumerate(table['header']):
        data[col] = list(zip(*table['rows']))[i]
    df = pd.DataFrame(data)
    print("[Table]")
    print(df)

    print("\n[Question]")
    print(question)

    print("\n[Answers]")
    for i, ans in enumerate(answers):
        print(ans)

    print("\n[Prediction]")
    print(log["prediction"], end='\t')
    if log["accuracy"] == 0:
        print("❌")
    else:
        print("✅")

def merge_items(l, p, o):
    return {
        "id": l["id"],
        "original_id": l["original_id"],
        "perturbation": l["perturbation_type"],
        "answer": l["answers"][0],
        "llama_pred": l["prediction"],
        "llama_acc": l["accuracy"],
        "pine_pred": p["prediction"],
        "pine_acc": p["accuracy"],
        "ours_pred": o["prediction"],
        "ours_acc": o["accuracy"],
        "table_len": len(l["table"]["rows"]),
    }

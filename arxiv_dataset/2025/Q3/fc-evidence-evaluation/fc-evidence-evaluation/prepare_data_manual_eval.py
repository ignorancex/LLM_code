import json
import random

import pandas as pd

averitec_test_path = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_test.json"
averitec_model_pred_path = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/baseline_pred_averitec_test.json"
_OUTPUT_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_manual_eval_with_label.csv"

_SAMPLE = {
    "id": None,
    "claim": None,
    "gold label": None,
    "reference evidence": None,
    "predicted evidence": None,
}
_QA_TEMPLATE = "Question: {}\nAnswer: {}\n"

_LABEL_MAPPING = {
    "Supported": "supported",
    "Refuted": "refuted",
    "Not Enough Evidence": "not enough information",
}
_INCLUDE_LABELS = True


def _format_evidence(questions) -> str:
    # load file
    qa_text = []
    qs = set()
    for qa in questions:
        if qa["question"].lower().strip() in qs:
            continue
        qs.add(qa["question"].lower().strip())
        answer = ""
        for a in qa["answers"]:
            answer += a["answer"]
            if "answer_type" in a and a["answer_type"] == "Boolean":
                answer += (" " + a["boolean_explanation"] + ". ")

        qa_template = _QA_TEMPLATE.format(qa["question"], answer)
        qa_text.append(qa_template)

    return "\n".join(qa_text)


def main():
    # load test file
    with open(averitec_test_path, "r", encoding="utf-8") as file:
        averitec_test = json.load(file)

    with open(averitec_model_pred_path, "r", encoding="utf-8") as file:
        averitec_model_pred = json.load(file)

    # iterate over averitec test
    counter = 0
    entries = []
    for i, entry in enumerate(averitec_test):
        if entry["label"] == "Conflicting Evidence/Cherrypicking":
            continue
        sample = _SAMPLE.copy()
        sample["id"] = counter
        sample["claim"] = entry["claim"]
        sample["reference evidence"] = _format_evidence(entry["questions"])
        sample["predicted evidence"] = _format_evidence(averitec_model_pred[i]["questions"])
        if _INCLUDE_LABELS:
            sample["gold label"] = _LABEL_MAPPING[entry["label"]]
        counter += 1
        entries.append(sample)

    # save as csv
    random.seed(10)
    entries_subset = random.sample(entries, 100)

    df = pd.DataFrame(entries_subset)
    df.to_csv(
        )
    print("Done!")


if __name__ == '__main__':
    main()

"""
mkdir -p misc/
wget https://raw.githubusercontent.com/FreedomIntelligence/HuatuoGPT-o1/refs/heads/main/evaluation/data/eval_data.json -O misc/eval_data.json

REPO_TYPE=dataset

REPO_URL_LIST=(
ReasoningEval/Benchmark_LastHumanity
ReasoningEval/Benchmark_Lancet
ReasoningEval/Benchmark_MedXpertQA
ReasoningEval/Benchmark_NEJM
ReasoningEval/Benchmark_MedBullets
)

for REPO_URL in ${REPO_URL_LIST[@]}; do
    LOCAL_DIR=misc/$REPO_URL

    mkdir -p $LOCAL_DIR
    huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}
done


The original datasets

- misc/ReasoningEval/Benchmark_Lancet/midium.jsonl
- misc/ReasoningEval/Benchmark_LastHumanity/midium.jsonl
- misc/ReasoningEval/Benchmark_MedBullets/medbullets_op4.jsonl
- misc/ReasoningEval/Benchmark_MedBullets/medbullets_op5.jsonl
- misc/ReasoningEval/Benchmark_MedXpertQA/midium.jsonl
- misc/ReasoningEval/Benchmark_NEJM/midium.jsonl
"""

import json
import pprint
from pathlib import Path

import datasets


def load_huatuo_eval():
    json_path = Path("misc/eval_data.json")
    with open(json_path, "r") as f:
        eval_data = json.load(f)
    return eval_data


def load_lancet():
    json_path = Path("misc/ReasoningEval/Benchmark_Lancet/midium.jsonl")
    with open(json_path, "r") as f:
        eval_data = [json.loads(line) for line in f]

    # NOTE: already fixed in the source
    # fix wrong answer
    # sample_to_be_correct = {
    #     "idx": 185,
    #     "question": "A mutation in one of the following genes might result in a rare metabolic disorder characterised by hyperphosphataemia and ectopic calcifications in periarticular soft tissues in the hip region. Which one is it?",
    #     "answer": "FGF23",
    #     "options": {"A": "PHEX", "B": "FGF23", "C": "CASR", "D": "SOST"},
    #     "answer_idx": "B",
    # }
    # for data in eval_data:
    #     if (
    #         data["idx"] == sample_to_be_correct["idx"]
    #         and data["question"].strip() == sample_to_be_correct["question"].strip()
    #         and data["options"] == sample_to_be_correct["options"]
    #     ):
    #         data["answer"] = sample_to_be_correct["answer"]
    #         data["answer_idx"] = sample_to_be_correct["answer_idx"]

    return {"Lancet": eval_data}


def load_humanity_last_exam():
    json_path = Path("misc/ReasoningEval/Benchmark_LastHumanity/midium.jsonl")
    with open(json_path, "r") as f:
        eval_data = [json.loads(line) for line in f]
    # fix answer value not match options
    for data in eval_data:
        # "C. Text\n" -> "Text"
        data["answer"] = data["answer"].strip().split(". ")[1]

    # NOTE: fix answer string, some answers only has the prefix
    # The question and options are in one string, use rule-based method
    # to extract the answer may lead to such mismatches.
    for data in eval_data:
        options = data["options"]
        answer = data["answer"]
        answer_idx = data["answer_idx"]
        if options[answer_idx] != answer:
            if options[answer_idx].startswith(answer):
                data["answer"] = options[answer_idx]

    return {"HumanityLastExam": eval_data}


def load_medbullets_op4():
    json_path = Path("misc/ReasoningEval/Benchmark_MedBullets/medbullets_op4.jsonl")
    with open(json_path, "r") as f:
        eval_data = [json.loads(line) for line in f]
    return {"MedBullets_op4": eval_data}


def load_medbullets_op5():
    json_path = Path("misc/ReasoningEval/Benchmark_MedBullets/medbullets_op5.jsonl")
    with open(json_path, "r") as f:
        eval_data = [json.loads(line) for line in f]
    return {"MedBullets_op5": eval_data}


def load_medxpertqa():
    json_path = Path("misc/ReasoningEval/Benchmark_MedXpertQA/midium.jsonl")
    with open(json_path, "r") as f:
        eval_data = [json.loads(line) for line in f]
    return {"MedXpertQA": eval_data}


def load_nejm():
    json_path = Path("misc/ReasoningEval/Benchmark_NEJM/midium.jsonl")
    with open(json_path, "r") as f:
        eval_data = [json.loads(line) for line in f]
    return {"NEJM": eval_data}


LOAD_FUNCTIONS = {
    "huatuo": load_huatuo_eval,
    "lancet": load_lancet,
    "humanity_last_exam": load_humanity_last_exam,
    "medbullets_op4": load_medbullets_op4,
    "medbullets_op5": load_medbullets_op5,
    "medxpertqa": load_medxpertqa,
    "nejm": load_nejm,
}

KEPT_KEYS = ["question", "options", "answer", "answer_idx"]


def main():
    data_dict = {}
    for key, load_func in LOAD_FUNCTIONS.items():
        data_dict.update(load_func())

    num_samples = {k: len(v) for k, v in data_dict.items()}
    print(f"Num samples (raw): {pprint.pformat(num_samples)}")

    check_answer_options_match(data_dict)
    strip_string(data_dict)
    check_answer_options_match(data_dict, True)

    num_samples = {k: len(v) for k, v in data_dict.items()}
    print(f"Num samples: {pprint.pformat(num_samples)}")

    sub_data_dict = {}
    num_sub_samples = 3
    for key, data in data_dict.items():
        sub_data_dict[key] = data[:num_sub_samples]
    output_sub_data_json_path = Path("misc/sub_eval_data.json")
    with open(output_sub_data_json_path, "w") as f:
        json.dump(sub_data_dict, f, indent=2)
    print(f"Saved sub eval data to {output_sub_data_json_path}")

    for data in data_dict.values():
        for sample in data:
            for key in list(sample.keys()):
                if key not in KEPT_KEYS:
                    del sample[key]

    output_eval_data_json_path = Path("misc/m1_eval_data.json")
    with open(output_eval_data_json_path, "w") as f:
        json.dump(data_dict, f, indent=2)
    print(f"Saved eval data to {output_eval_data_json_path}")


def check_answer_options_match(eval_data, remove_not_match=False):
    not_matches = {}
    for key, data in eval_data.items():
        not_matches[key] = []
        for i, sample in enumerate(data):
            answer = sample["answer"]
            answer_idx = sample["answer_idx"]

            option_answer = sample["options"][answer_idx]
            if answer != option_answer:
                not_matches[key].append(sample)
                if remove_not_match:
                    data.remove(sample)
    not_matches_path = Path("misc/not_matches.json")
    with open(not_matches_path, "w") as f:
        json.dump(not_matches, f, indent=2)
    num_not_matches = {k: len(v) for k, v in not_matches.items()}
    print(f"Num not matches: {pprint.pformat(num_not_matches)}")
    print(f"Saved not matches to {not_matches_path}")


def strip_string(data_dict):
    for key, data in data_dict.items():
        for sample in data:
            sample["question"] = sample["question"].strip()
            sample["options"] = {
                idx.strip(): answer.strip() for idx, answer in sample["options"].items()
            }

            sample["answer"] = sample["answer"].strip()
            sample["answer_idx"] = sample["answer_idx"].strip()


if __name__ == "__main__":
    main()

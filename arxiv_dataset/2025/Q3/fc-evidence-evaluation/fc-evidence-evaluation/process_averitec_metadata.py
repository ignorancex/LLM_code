import argparse
import json

import properties
import utils

parser = argparse.ArgumentParser(
    description='Prompt Scorer OpenAI arguments'
)
parser.add_argument(
    '--dataset_file_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_w_metadata.json",
    help='Path to averitec data containing metadata, e.g. about data added, edited in different phases.'
)
parser.add_argument(
    '--dataset_before_p4_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_w_metadata_before_p4.jsonl",
    help='Subset of entries which requires adding and/or editing of evidence in phase P4.'
)
parser.add_argument(
    '--dataset_after_p4_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_w_metadata_after_p4.jsonl",
    help='Subset of entries which requires adding and/or editing of evidence in phase P4, AFTER changes.'
)

args = parser.parse_args()
_DATASET_FILE_PATH = args.dataset_file_path
_BEFORE_P4_FILE_PATH = args.dataset_before_p4_path
_AFTER_P4_FILE_PATH = args.dataset_after_p4_path


def map_answers(answers: list):
    mapped_answers = []
    for answer in answers:
        if "boolean_explanation" in answer:
            mapped_answers.append(
                properties.AveritecAnswer(answer["answer"], answer["answer_type"], answer["boolean_explanation"]))
        else:
            mapped_answers.append(properties.AveritecAnswer(answer["answer"], answer["answer_type"]))
    return mapped_answers


def map_evidences(dataset_entry: dict, key: str):
    return [properties.AveritecQA(e["question"], map_answers(e["answers"])) for e in dataset_entry[key]]


def is_not_required_entry(entry: dict) -> bool:
    if "phase_5_label" in entry and (
            # only samples interesting for us whose evidence had limitations but label didn't change
            entry["phase_5_label"] != entry["phase_3_label"] or entry["phase_5_label"] in ["",
                                                                                           "Conflicting "
                                                                                           "Evidence/Cherrypicking"]):
        return True
    elif "phase_5_label" not in entry and entry["phase_3_label"] in ["", "Conflicting Evidence/Cherrypicking"]:
        return True
    return False


def extract_entries_requiring_p4_adjustments(dataset: list):
    """Extracts samples that require reannotation and whose labels didn't change after annotation."""
    subset = []
    for entry in dataset:
        if is_not_required_entry(entry):
            continue
        if 'p2_with_p4_edit_questions' in entry or 'only_p4_questions' in entry:
            # prepare entry before adding
            evidences = map_evidences(entry, "only_p2_questions")
            subset.append(properties.AveritecEntry(
                entry["claim"], entry["phase_3_label"], entry["justification_p5"], evidences
            ))
    return subset


def extract_p4_adjusted_entries(dataset: list):
    subset = []
    for entry in dataset:
        if is_not_required_entry(entry):
            continue
        if 'p2_with_p4_edit_questions' in entry or 'only_p4_questions' in entry:
            # prepare entry before adding
            evidences = []
            if 'p2_with_p4_edit_questions' in entry:
                evidences.extend(map_evidences(entry, "p2_with_p4_edit_questions"))
            if 'only_p4_questions' in entry:
                evidences.extend(map_evidences(entry, "only_p4_questions"))
            # TODO do we still need only_p2_questions?
            subset.append(properties.AveritecEntry(
                entry["claim"], entry["phase_3_label"], entry["justification_p5"], evidences
            ))

    return subset


def main():
    print("Loading test data..")
    with open(_DATASET_FILE_PATH, encoding="utf-8") as file:
        dataset = json.load(file)

    subset_req_p4_changes = extract_entries_requiring_p4_adjustments(dataset)
    subset_after_p4_changes = extract_p4_adjusted_entries(dataset)
    print("{} vs. {}".format(len(subset_req_p4_changes), len(subset_after_p4_changes)))

    utils.save_jsonl_file(subset_req_p4_changes, _BEFORE_P4_FILE_PATH)
    utils.save_jsonl_file(subset_after_p4_changes, _AFTER_P4_FILE_PATH)


if __name__ == '__main__':
    main()

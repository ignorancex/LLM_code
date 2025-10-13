"""
Loads Averitec Shared task samples manually annotated.
Prints gold samples for evaluation.
Post-processes data to allow calculation of correlation with evaluation scorers.
"""
import json

import pandas as pd

PATH_ANNOTATIONS = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec_shared_task/AVERITEC_FC_Evidence_Evaluation_Responses.xlsx"
PATH_AVERITEC_SHARED_TASK = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_shared_task_test.json"
PATH_EVAL_DATA = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec_shared_task/manual_eval_selection.json"
OUTPUT_PATH_NOGOLD_ANNOTATIONS = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec_shared_task/AVERITEC_FC_Evidence_Evaluation_Responses_no_gold.xlsx"
PATH_GOLD_SAMPLES = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec_shared_task/gold_samples_output.txt"

_GOLD_ANNOTATION_IDS = [333, 446, 844, 1161, 1641]
_QA_TEMPLATE = "Question: {}\nAnswer: {}\n"


# functions
def get_entry_eval_data(eval_id, testset: list):
    """
    Given ID, extracts sample from Averitec shared task testset.
    """
    for e in testset:
        if int(e['claim_id']) == int(eval_id):
            return e
    return None


def _format_evidence_ref(evidences: list) -> str:
    """
    Formats a list of (reference) evidence to match the format required for correlation calculation.
    """
    qa_text = []
    qs = set()
    for qa in evidences:
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


def _format_evidence_pred(questions) -> str:
    """
    Formats a list of (predicted) evidence to match the format required for correlation calculation.
    """
    qa_text = []
    qs = set()
    for qa in questions:
        if qa["question"].lower().strip() in qs:
            continue
        qs.add(qa["question"].lower().strip())
        qa_template = _QA_TEMPLATE.format(qa["question"], qa["answer"])
        qa_text.append(qa_template)

    return "\n".join(qa_text)


def _label_matching(gold, annotated):
    annotated = annotated.replace("information", "evidence")
    return int(gold.lower().strip() == annotated.lower().strip())


def main():
    # load relevant data
    annotations = pd.read_excel(PATH_ANNOTATIONS, header=0)
    with open(PATH_AVERITEC_SHARED_TASK, "r") as file:
        shared_task_data = json.load(file)
    with open(PATH_EVAL_DATA, "r") as file:
        eval_data = json.load(file)

    # print for each team number of annotations
    annotations_teams = annotations['Team'].value_counts()
    print("Number of annotations per team: {}".format(annotations_teams))

    annotations_no_gold = pd.DataFrame(columns=annotations.columns)
    count_evidence_issues = 0
    for i, row in annotations.iterrows():
        if row['id'] not in _GOLD_ANNOTATION_IDS:
            # samples not belonging to gold-labelled ones
            if "Yes," in row[
                '1. Does the predicted evidence contain any of the following three major errors? If yes, which of the ' \
                'following holds for the predicted evidence?']:
                # filter out annotations which flag issues in predicted evidence
                count_evidence_issues += 1
                continue

            # add missing data, i.e., gold label, reference evidence, predicted evidence, predicted label
            entry = shared_task_data[int(row['id'])]
            row['gold label'] = entry['label']
            row['reference evidence'] = _format_evidence_ref(entry['questions'])

            eval_entry = get_entry_eval_data(eval_id=row['id'], testset=eval_data)
            row['predicted evidence'] = _format_evidence_pred(eval_entry['predicted_evidence'])
            row['pred_label'] = eval_entry['pred_label']

            row['label_majority'] = row['label_majority'][2:]

            # add verdict_agreement and nei_disagreement flags
            row["verdict_agreement"] = _label_matching(row['gold label'], row['label_majority'])
            # row["nei_disagreement"]

            annotations_no_gold = annotations_no_gold.append(row, ignore_index=True)

        # save gold-labelled samples in a file to manually check and assess annotations
        with open(PATH_GOLD_SAMPLES, "a") as file:
            file.write("Team: {}\n".format(row['Team']))
            file.write("ID: {}\n".format(row['id']))
            file.write("Claim: {}\n".format(row['claim']))
            file.write("Errors: {}\n".format(row[
                                                 '1. Does the predicted evidence contain any of the following three major errors? If yes, which of the following holds for the predicted evidence?']))
            file.write("Label: {}\n".format(row["label_majority"]))
            file.write("Pred Evidence: {}\n".format(row['predicted evidence']))

            file.write("semantic_coverage: {}\n".format(row['semantic_coverage']))
            file.write("coherence: {}\n".format(row['coherence']))
            file.write("repetition: {}\n".format(row['redundancy']))
            file.write("consistency: {}\n".format(row['consistency']))
            file.write("relevance: {}\n".format(row['relevance']))

            file.write("------------------------------\n")

    # save non-gold samples separately for eval purposes
    print("Len of filtered dataframe: {} (prev {})".format(len(annotations_no_gold), len(annotations)))
    annotations_no_gold.to_excel(OUTPUT_PATH_NOGOLD_ANNOTATIONS)

    print("Number of entries filtered due to issues in evidence: {}".format(count_evidence_issues))


if __name__ == "__main__":
    main()

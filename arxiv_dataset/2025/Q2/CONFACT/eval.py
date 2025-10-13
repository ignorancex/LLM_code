import os
import re
import csv
from typing import List
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import load_json
import glob

def label_mapping(source_data):
    label_mapping = {
            "Supported": "yes",
            "Refuted": "no",
        }
    return [label_mapping.get(entity['ground_truth'].strip(), "unsure") for entity in source_data]

def extract_predictions(source_data):
    return [entity['prediction'] for entity in source_data]

class EvaluationEngine:
    def __init__(self):
        pass

    def extract_final_answer(self, text: str) -> str:
        # extract 'yes' or 'no' from a text. Returns empty string if not found.
        pattern = r"\b(yes|no)\b"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        else:
            return ""

    def evaluate(self, predictions: List[str], ground_truth: List[str]):

        em = 0  # exact matches
        true_positive = 0
        false_positive = 0
        false_negative = 0

        all_predictions = []
        all_ground_truth = []

        for idx, prediction in enumerate(predictions):
            predicted_answer = self.extract_final_answer(prediction)
            actual_answer = ground_truth[idx]

            # Convert 'yes'/'no' to 1/0 for metric calculation
            if "yes" in actual_answer.lower():
                actual_answer_bin = 1
            elif "no" in actual_answer.lower():
                actual_answer_bin = 0
            else:
                print("Ground truth not recognized as 'yes' or 'no':", actual_answer)
                continue

            if "yes" in predicted_answer.lower():
                predicted_answer_bin = 1
            elif "no" in predicted_answer.lower():
                predicted_answer_bin = 0
            else:
                print("Prediction not recognized as 'yes' or 'no':", predicted_answer)
                continue

            # Exact Match (EM for Accuracy)
            if actual_answer_bin == predicted_answer_bin:
                em += 1

            # For precision & recall
            if actual_answer_bin == predicted_answer_bin:
                true_positive += 1
            else:
                if predicted_answer_bin == 1:
                    false_positive += 1
                if actual_answer_bin == 1:
                    false_negative += 1

            all_predictions.append(predicted_answer_bin)
            all_ground_truth.append(actual_answer_bin)

        # Accuracy
        acc = em / len(predictions) if len(predictions) > 0 else 0.0

        # Precision, Recall, F1
        precision = precision_score(all_ground_truth, all_predictions, average='binary', zero_division=0)
        recall = recall_score(all_ground_truth, all_predictions, average='binary', zero_division=0)
        f1 = f1_score(all_ground_truth, all_predictions, average='binary', zero_division=0)

        weighted_f1 = f1_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
        macro_f1 = f1_score(all_ground_truth, all_predictions, average='macro', zero_division=0)
    
        return acc, precision, recall, f1, weighted_f1, macro_f1


    def evaluate_and_log(
        self, 
        acc: float, precision: float, recall: float, f1: float, weighted_f1: float, macro_f1: float,
        file_name: str, 
        csv_file: str
    ):
        directory = os.path.dirname(csv_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_exists = os.path.exists(csv_file)
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # If file is new, write the header
            if not file_exists:
                writer.writerow(['file_name', 'accuracy', 'precision', 'recall', 'f1_score', 'weighted_f1', 'macro_f1'])
            writer.writerow([file_name, acc, precision, recall, f1, weighted_f1, macro_f1])

def main():

    parser = argparse.ArgumentParser(description="Evaluation of results")
    parser.add_argument("--folder", type=str, default= './results/results_all_media', help="path to the results folder")
    args = parser.parse_args()

    folder_path = args.folder
    results_files = glob.glob(f"{folder_path}/*.json")

    for result_file in results_files:

        results = load_json(result_file)

        ground_truths = label_mapping(results)
        predictions = extract_predictions(results)

        eval_engine = EvaluationEngine()
        acc, precision, recall, f1, weighted_f1, macro_f1 = eval_engine.evaluate(predictions, ground_truths)
        
        file_name = os.path.basename(result_file)

        eval_engine.evaluate_and_log(
            acc, precision, recall, f1, weighted_f1, macro_f1,
            file_name = file_name,
            csv_file = os.path.join(folder_path,'evaluation_results.csv')
        )
        
if __name__ == "__main__":
    main()


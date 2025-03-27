import string
import re
import json
import sys
import os
import argparse
import logging
from collections import Counter

import numpy as np
from rouge import rouge_scorer
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)

class GPTTokenizer:
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2", max_length=1e5)

    def tokenize(self, s):
        tokens = self.gpt_tokenizer.tokenize(s)
        # GPT2 uses Byte-level BPE, which will include space as part of the word. 
        # But for the first word of a sentence, there is no space before it. 
        # So, we remove all the added spaces ("Ġ"). 
        tokens = [t.lstrip("Ġ") for t in tokens]
        return tokens

xlingual_tokenizer = GPTTokenizer()


# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match_score(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def rouge1_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rouge1'], tokenizer=xlingual_tokenizer)
    else:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rouge1"].fmeasure


def rougeL_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 
    else:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_metrics(predictions, references, xlingual=False):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match, rouge1, rougeL = 0, 0, 0
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rouge1 += metric_max_over_ground_truths(
            rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rougeL += metric_max_over_ground_truths(
            rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    exact_match = 100.0 * exact_match / len(references)
    rouge1 = 100.0 * rouge1 / len(references)
    rougeL = 100.0 * rougeL / len(references)
    metrics = {"exact_match": exact_match, "rouge1": rouge1, "rougeL": rougeL}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


def compute_grouped_metrics(predictions, references, groups, xlingual=False):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))

    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics_with_jackknife_variance(task_predictions, task_references, xlingual=xlingual) # compute_metrics replaced

        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results


def compute_metrics_with_bootstrap_variance(predictions, references, xlingual=False):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match_list, rouge1_list, rougeL_list = [], [], []
    
    num_resample = 500 # how many times we resample from the sample
    for _ in range(num_resample):
        # Generate a set of random indices
        num_samples = 50 # sample size
        random_indices = np.random.choice(len(predictions), size=num_samples, replace=True)
        
        # Use the same random indices to resample both pred and ref
        sampled_predictions = [predictions[i] for i in random_indices]
        sampled_references = [references[i] for i in random_indices]

        exact_match, rouge1, rougeL = 0, 0, 0
        
        for pred, gold in zip(sampled_predictions, sampled_references):
            assert isinstance(gold, list)
            exact_match += metric_max_over_ground_truths(
                exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
            )
            rouge1 += metric_max_over_ground_truths(
                rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
            )
            rougeL += metric_max_over_ground_truths(
                rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
            )
        exact_match = 100.0 * exact_match / num_samples
        rouge1 = 100.0 * rouge1 / num_samples
        rougeL = 100.0 * rougeL / num_samples

        exact_match_list.append(exact_match)
        rouge1_list.append(rouge1)
        rougeL_list.append(rougeL)

    # Calculate mean and variance
    exact_match_mean = round(np.mean(exact_match_list), 4)
    exact_match_var = round(np.var(exact_match_list), 4)

    rouge1_mean = round(np.mean(rouge1_list), 4)
    rouge1_var = round(np.var(rouge1_list), 4)

    rougeL_mean = round(np.mean(rougeL_list), 4)
    rougeL_var = round(np.var(rougeL_list), 4)

    # Create the metrics dictionary
    metrics = {
        "exact_match": (exact_match_mean, exact_match_var),
        "rouge1": (rouge1_mean, rouge1_var),
        "rougeL": (rougeL_mean, rougeL_var)
    }

    return metrics

def compute_metrics_with_jackknife_variance(predictions, references, xlingual=False):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match_list, rouge1_list, rougeL_list = [], [], []
    
    num_samples = len(predictions)
    
    for i in range(num_samples):
        # Omit one sample at a time
        omitted_predictions = predictions[:i] + predictions[i+1:]
        omitted_references = references[:i] + references[i+1:]

        exact_match, rouge1, rougeL = 0, 0, 0
        
        for pred, gold in zip(omitted_predictions, omitted_references):
            assert isinstance(gold, list)
            exact_match += metric_max_over_ground_truths(
                exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
            )
            rouge1 += metric_max_over_ground_truths(
                rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
            )
            rougeL += metric_max_over_ground_truths(
                rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
            )
        exact_match = 100.0 * exact_match / (num_samples - 1)  # Subtract one for the omitted sample
        rouge1 = 100.0 * rouge1 / (num_samples - 1)
        rougeL = 100.0 * rougeL / (num_samples - 1)

        exact_match_list.append(exact_match)
        rouge1_list.append(rouge1)
        rougeL_list.append(rougeL)

    # Calculate mean and variance
    exact_match_mean = round(np.mean(exact_match_list), 4)
    exact_match_var = round(np.var(exact_match_list), 4)

    rouge1_mean = round(np.mean(rouge1_list), 4)
    rouge1_var = round(np.var(rouge1_list), 4)

    rougeL_mean = round(np.mean(rougeL_list), 4)
    rougeL_var = round(np.var(rougeL_list), 4)

    # Create the metrics dictionary
    metrics = {
        "exact_match": (exact_match_mean,exact_match_var),
        "rouge1": (rouge1_mean,rouge1_var),
        "rougeL": (rougeL_mean,rougeL_var)
    }

    return metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions file.")
    parser.add_argument("--track", choices=["default", "xlingual"], default="default", 
        help="default track or xlingual track. For xlingual, we need to use a different tokenizer."
    )
    parser.add_argument("--compute_per_category_metrics", action="store_true", help="Compute metrics on every evaluation category.")
    parser.add_argument("--compute_per_task_metrics", action="store_true", help="Compute metrics on every evaluation task.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.predictions) as fin:
        examples = [json.loads(l) for l in fin]

    predictions = [e["Prediction"] for e in examples]
    references = [e["Instance"]["output"] for e in examples]
  
    tasks = []
    for e in examples:
        if e["Task"] == "task121_atomic_question_rewriting":
            e["Task"] = "task121_zest_question_rewriting"
        tasks.append(e["Task"])

    # results = compute_metrics(predictions, references, xlingual=args.track == "xlingual")
    # print("======== Overall Metrics ========")
    # print("all_rougeL", results["rougeL"])
    # print("all_EM", results["exact_match"])
    # print()
    
    category_metrics = [
        ("Textual Entailment", "exact_match"),
        ("Cause Effect Classification", "exact_match"),
        ("Coreference Resolution", "exact_match"),
        ("Dialogue Act Recognition", "exact_match"),
        ("Answerability Classification", "exact_match"),
        ("Word Analogy", "exact_match"),
        ("Overlap Extraction", "rougeL"),
        ("Keyword Tagging", "rougeL"),
        ("Question Rewriting", "rougeL"),
        ("Title Generation", "rougeL"),
        ("Data to Text", "rougeL"),
        ("Grammar Error Correction", "rougeL"),
        ("Question Answering", "rougeL"),
        ("Text Categorization", "exact_match"),
        ("Commonsense Classification", "exact_match"),
        ("Sentiment Analysis", "exact_match"),
        ("Text Matching", "exact_match"),
        ("Grammar Error Detection", "exact_match")
    ]
    category_metrics = {"_".join(category.lower().split()): metric for category, metric in category_metrics}

    if args.compute_per_category_metrics:
        # print("======== Metrics per category ========")
        
        categories = []
        for e in examples:
            category = e["Categories"][0].lower().split()
            category = "_".join(category)
            categories.append(category)

        task_category = {}
        for task in set(tasks):
            with open(os.path.join("../src/data_generator/processed_tasks/", task+"_baseline.json")) as fin:
                task_data = json.load(fin)
                task_category[task] = "_".join(task_data["Categories"][0].lower().split())

        # results.update(compute_grouped_metrics(predictions, references, categories, xlingual=args.track=="xlingual"))
        
        # for category, metric in category_metrics.items():
        #     # category = "_".join(category.lower().split())
        #     if f"{metric}_for_{category}" in results:
        #         print(f"{metric}_for_{category}", results[f"{metric}_for_{category}"])
        # print()
            
    if args.compute_per_task_metrics:
        results_by_task = compute_grouped_metrics(predictions, references, tasks, xlingual=args.track=="xlingual")
    
        # for task in sorted(list(set(tasks))):
          
        #     category = task_category[task]
        #     metric = category_metrics[category]
        #     print(task, results_by_task[f"{metric}_for_{task}"])
        # print()

        # Define the desired order of task names
        desired_order = [
            "task1344_glue_entailment_classification",
            "task1645_medical_question_pair_dataset_text_classification",
            "task843_financial_phrasebank_classification",
            "task512_twitter_emotion_classification",
            "task1346_glue_cola_grammatical_correctness_classification",
            "task379_agnews_topic_classification",
            "task828_copa_commonsense_cause_effect",
            "task116_com2sense_commonsense_reasoning",
            "task1564_triviaqa_answer_generation",
            "task835_mathdataset_answer_generation",
            "task284_imdb_classification",
        ]

        # Extract unique task names and sort them according to the desired order
        unique_tasks = sorted(set(tasks), key=lambda x: desired_order.index(x))

        metr = []
        # Iterate through the sorted task names and print the corresponding metrics
        print("======== Metrics per task ========")
        for task in unique_tasks:
            category = task_category[task]
            metric = category_metrics[category]
            
            metr.append(results_by_task[f"{metric}_for_{task}"])
            print(task, results_by_task[f"{metric}_for_{task}"])

        metr = "[" + ";".join(map(str, metr)) + "]"
        print(metr)
import argparse
import json
import os
from typing import Dict, List

import evaluate
import nltk
import numpy as np
import spacy
from datasets import load_dataset
from other_utils import (
    filter_by_num_sentences,
    flatten_squality,
    get_summaries_from_predictions_file,
    process_summscreen,
)


def load_reference_summaries(
    dataset_name, dataset_cache_dir, min_num_sentences: int = 1
):
    if dataset_name == "multinews":
        pass
    elif dataset_name == "summscreen":
        dataset = load_dataset(
            "YuanPJ/summ_screen", "fd", split="test", cache_dir=dataset_cache_dir
        )
        dataset = process_summscreen(dataset)
        if min_num_sentences > 1:
            nlp = spacy.load("en_core_web_md")
            dataset = dataset.filter(
                filter_by_num_sentences,
                fn_kwargs={"nlp": nlp, "min_num_sentences": min_num_sentences},
            )
        reference_summaries = dataset["summary"]
    elif dataset_name == "squality":
        dataset = load_dataset(
            "pszemraj/SQuALITY-v1.3", split="test", cache_dir=dataset_cache_dir
        )
        dataset = flatten_squality(dataset)
        reference_summaries = dataset["summary"]
    else:
        raise ValueError("Dataset should be multinews, summscreen or squality.")

    return reference_summaries


def compute_rouge(
    reference_summaries: List[str],
    predicted_summaries: List[str],
    num_reference_summaries: int,
) -> Dict:
    # reference_summaries has the form: [first set of ref summaries for all data points, second set . . ., third set and so on]
    # we need to transpose it
    dataset_size = len(reference_summaries) // num_reference_summaries
    reference_summaries = [
        reference_summaries[i * dataset_size : (i + 1) * dataset_size]
        for i in range(0, num_reference_summaries)
    ]
    # transpose it
    reference_summaries = list(map(list, zip(*reference_summaries)))
    # flatten it
    reference_summaries = [x.strip() for dl in reference_summaries for x in dl]
    predicted_summaries = list(np.repeat(predicted_summaries, num_reference_summaries))
    predicted_summaries = [str(s) for s in predicted_summaries]

    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in predicted_summaries]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in reference_summaries]
    rouge = evaluate.load("rouge")
    non_agg_results = rouge.compute(
        predictions=preds,
        references=labels,
        use_stemmer=True,
        use_aggregator=False,
    )

    agg_results = {}
    for metric in non_agg_results.keys():
        # compute and add max
        values = non_agg_results[metric]
        agg_results[metric] = np.mean(values)
        agg_results[metric + "_sd"] = np.std(values)
        values = [
            max(values[i : i + num_reference_summaries])
            for i in range(0, len(values), num_reference_summaries)
        ]
        agg_results[metric + "_max"] = np.mean(values)
        agg_results[metric + "_max_sd"] = np.std(values)

    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    # agg_results["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in agg_results.items()}


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="squality")
    parser.add_argument("--predictions_file", type=str, default=None)
    parser.add_argument("--num_reference_summaries", type=int, default=4)
    parser.add_argument("--results_file_path", type=str, default=None)
    parser.add_argument(
        "--dataset_cache_dir", type=str, default="/home/ubuntu/hf_data_cache"
    )
    parser.add_argument(
        "--min_num_sentences",
        type=int,
        default=1,
        help="minimum number of sentences in the reference summary to retain that data point",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_cmd_args()
    # read the summaries file
    reference_summaries = load_reference_summaries(
        args.dataset_name,
        args.dataset_cache_dir,
        min_num_sentences=args.min_num_sentences,
    )
    predicted_summaries = get_summaries_from_predictions_file(args.predictions_file)

    assert len(reference_summaries) % args.num_reference_summaries == 0
    assert len(predicted_summaries) == (
        len(reference_summaries) // args.num_reference_summaries
    )
    results = compute_rouge(
        reference_summaries, predicted_summaries, args.num_reference_summaries
    )
    # write results to a file
    with open(os.path.join(args.results_file_path, "rouge.json"), "w") as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    main()

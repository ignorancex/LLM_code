import argparse
import json

import datasets
import spacy
from other_utils import get_summaries_from_predictions_file


def extract_truncated_document_from_text_column(example):
    text = example["text"]
    # find index of Text:
    text_index = text.find("Text:")
    # find first index of <|end|> after text_index
    end_index = text.find("<|end|>", text_index)
    # extract document
    truncated_document = text[text_index + len("Text:") : end_index].strip()
    return {"truncated_document": truncated_document}


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="path to hf dataset"
    )
    parser.add_argument(
        "--column_name",
        type=str,
        choices=["text", "document"],
        default="text",
        help="column name to use from hf dataset",
    )
    parser.add_argument("--predictions_file", type=str, required=True)
    parser.add_argument("--finesure_format_file", type=str, required=True)
    parser.add_argument(
        "--task", choices=["baseline", "e2e", "multitask"], required=True
    )
    parser.add_argument("--num_reference_summaries", type=int, default=4)
    parser.add_argument("--do_sample", action="store_true", default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # load dataset from disk
    dataset = datasets.load_from_disk(args.dataset_path)
    test_dataset = dataset["test"]
    if args.task == "multitask":
        # split dataset_split into two equal sized datasets: plans and summaries
        test_dataset = test_dataset.select(
            range(len(test_dataset) // 2, len(test_dataset))
        )

    if args.num_reference_summaries > 1:
        # multiple ref summaries and no sampling
        if not args.do_sample:
            dataset_size = len(test_dataset) // args.num_reference_summaries
            # select the first dataset_size data points from summary_dataset and plan_dataset
            test_dataset = test_dataset.select(range(dataset_size))

    if args.column_name == "document":
        transcripts = test_dataset["document"]
    elif args.column_name == "text":
        test_dataset = test_dataset.map(extract_truncated_document_from_text_column)
        transcripts = test_dataset["truncated_document"]
    else:
        raise ValueError(f"Invalid column name: {args.column_name}")
    predicted_summaries = get_summaries_from_predictions_file(args.predictions_file)
    assert len(predicted_summaries) == len(transcripts)
    nlp = spacy.load("en_core_web_md")
    with open(args.finesure_format_file, "w") as fhw:
        for i, predicted_summary in enumerate(predicted_summaries):
            # get sentences using spacy
            sentences = [sent.text.strip() for sent in nlp(predicted_summary).sents]
            output = {
                "doc_id": i,
                "split": "test",
                "model": "phi3",
                "transcript": transcripts[i],
                "summary": predicted_summary,
                "sentences": sentences,
            }
            # import ipdb
            # ipdb.set_trace(context=20)
            fhw.write(json.dumps(output) + "\n")


if __name__ == "__main__":
    main()

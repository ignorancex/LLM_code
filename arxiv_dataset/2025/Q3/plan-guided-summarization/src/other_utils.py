import argparse
import json

import spacy
from datasets import Dataset, concatenate_datasets, load_dataset


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def flatten_squality(squality_ds: Dataset):
    # there are 4 summaries for each document so we flatten the dataset so that we have four times the data.
    documents = squality_ds["document"]
    metadata = squality_ds["metadata"]
    questions = squality_ds["questions"]
    summary1 = [q[0]["responses"][0]["response_text"] for q in questions]
    summary2 = [q[0]["responses"][1]["response_text"] for q in questions]
    summary3 = [q[0]["responses"][2]["response_text"] for q in questions]
    summary4 = [q[0]["responses"][3]["response_text"] for q in questions]
    dataset1 = Dataset.from_dict({"document": documents, "metadata": metadata, "summary": summary1})
    dataset2 = Dataset.from_dict({"document": documents, "metadata": metadata, "summary": summary2})
    dataset3 = Dataset.from_dict({"document": documents, "metadata": metadata, "summary": summary3})
    dataset4 = Dataset.from_dict({"document": documents, "metadata": metadata, "summary": summary4})
    flattened_squality = concatenate_datasets([dataset1, dataset2, dataset3, dataset4])
    return flattened_squality


def create_metadata(example):
    metadata = {
        "File Name": example["File Name"],
        "Show Title": example["Show Title"],
        "Episode Number": example["Episode Number"],
        "Episode Title": example["Episode Title"],
        "Recap Author": example["Recap Author"],
        "Transcript Author": example["Transcript Author"],
    }
    return {"metadata": metadata}


def process_summscreen(summscreen_ds: Dataset):
    transcripts = ["\n".join(transcript) for transcript in summscreen_ds["Transcript"]]
    summscreen_ds = summscreen_ds.add_column("document", transcripts)
    # summscreen_ds["document"] = transcripts
    recaps = [recap[0] for recap in summscreen_ds["Recap"]]
    # summscreen_ds["summary"] = recaps
    summscreen_ds = summscreen_ds.add_column("summary", recaps)
    summscreen_ds = summscreen_ds.map(create_metadata, num_proc=8)
    # delete all columns except document, summary and metadata
    summscreen_ds = summscreen_ds.remove_columns(
        [
            "File Name",
            "Show Title",
            "Episode Number",
            "Episode Title",
            "Recap Author",
            "Transcript Author",
            "Transcript",
            "Recap",
        ]
    )
    return summscreen_ds


def extract_summary_from_txt(plan_summary):
    # replace <|assistant|>, <|end|> and <|endoftext|> with empty string
    plan_summary = plan_summary.replace("<|assistant|>", "")
    plan_summary = plan_summary.replace("<|end|>", "")
    plan_summary = plan_summary.replace("<|endoftext|>", "")
    if "<summary>" in plan_summary:
        plan_summ_split_idx = plan_summary.index("<summary>")
        split_idx = plan_summ_split_idx + len("<summary>")
        summary = plan_summary[split_idx:]
        # remove </summary> from summary if present
        summary = summary.replace("</summary>", "")
        if not summary.strip():  # The summary doesn't exist, just use plan
            summary = plan_summary
    else:
        # Cannot find the summary split, so we use the output as both summary and plan
        summary = plan_summary

    summary = summary.replace("<pad>", "")
    # summary = summary.replace("<plan>", "")
    summary = summary.replace("</s>", "")
    # summary = re.sub(r"[0-9]+\. ", "", summary)

    return summary.strip()


def get_summaries_from_predictions_file(predictions_file):
    summaries = []
    with open(predictions_file, "r") as fhr:
        for line in fhr:
            line = json.loads(line.strip())
            summaries.append(extract_summary_from_txt(line["output"]))
    return summaries


def filter_by_num_sentences(datum, nlp, min_num_sentences):
    summary = datum["summary"]
    # get sentences using spacy
    sentences = [sent.text for sent in nlp(summary).sents]
    if len(sentences) >= min_num_sentences:
        return True
    else:
        return False


def load_documents(dataset_name, min_num_sentences: int = 1):
    nlp = spacy.load("en_core_web_md")
    dataset_cache_dir = "/home/ec2-user/SageMaker/hf_data_cache"
    if dataset_name == "summscreen":
        dataset = load_dataset(
            "YuanPJ/summ_screen", "fd", split="test", cache_dir=dataset_cache_dir
        )
        dataset = process_summscreen(dataset)
        if min_num_sentences > 1:
            dataset = dataset.filter(
                filter_by_num_sentences,
                fn_kwargs={"nlp": nlp, "min_num_sentences": min_num_sentences},
            )
        documents = dataset["document"]
    elif dataset_name == "squality":
        dataset = load_dataset("pszemraj/SQuALITY-v1.3", split="test", cache_dir=dataset_cache_dir)
        dataset = flatten_squality(dataset)
        documents = dataset["document"]
    else:
        raise ValueError("Dataset should be multinews, summscreen or squality")

    return documents

import os
import random
import re

import datasets
import nltk
import numpy as np
import pandas as pd
import scipy
from nltk import word_tokenize

from properties import PseudoTrainedScorerDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import json
import sqlite3
import unicodedata
from typing import List, Tuple

import dacite
import evaluate
import torch
from torch.utils.data import DataLoader
from nltk.tokenize import sent_tokenize
import properties

metric = evaluate.load("f1")
metric_meteor = datasets.load_metric("meteor")
metric_rouge = datasets.load_metric("rouge")
metric_bleu = datasets.load_metric("bleu")

SAMPLE_DICT = {
    'claim': None,
    'evidence': None,
    'label': None,
}


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)


def get_random_entry(dataset: list, current_entry: dict) -> dict:
    """
    Returns a random entry from dataset 'dataset' which is different from 'current_entry'
    :param dataset:
    :param current_entry:
    :return:
    """
    while True:
        random_entry = random.choice(dataset)
        if random_entry['claim'] != current_entry['claim']:
            break

    return random_entry


def load_json_file(path: str):
    """Loads data from path."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json_file(data_sample, path: str):
    """Saves data in file at given path."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data_sample, file, indent=4)


def _load_fever_evidence(evidences: list, wiki_db):
    evidence_text = ""
    for e in evidences:
        try:
            evid = wiki_db[wiki_db['id'] == e[2]]["lines"].values[0]
            # wiki_db.execute("SELECT * FROM fever.wiki_pages WHERE id = (%s)", [unicodedata.normalize('NFD', e[2])])
            # for doc in wiki_db:
            #     # retrieve relevant sentence as evidence
            evidence_text += evid.split("\n")[e[3]].split("\t")[1]
        except Exception as e:
            print(e)
            continue
    return evidence_text


def read_fever_shared(file_path):
    claims = []
    evidences = []
    labels = []
    with open(file_path) as f:
        for line in f:
            line_loaded = json.loads(line)
            for entry in list(line_loaded[1].values()):
                claim = entry["claim"]
                label = properties.LABEL_DICT[properties.Label(entry["label"].lower())]
                for evidence_tuple in entry["evidence"]:
                    if len(evidence_tuple) == 3:
                        evidence = evidence_tuple[2]
                        labels.append(label)
                        claims.append(claim)
                        evidences.append(evidence)
                    else:
                        continue

    return claims, evidences, labels


def read_fever_base(file_path, wiki_db):
    claims = []
    evidences = []
    labels = []
    with open(file_path) as f:
        for entry in f:
            entry = json.loads(entry)
            claim = entry["claim"]
            label = properties.LABEL_DICT[properties.Label(entry["label"].lower())]
            for evidence in entry["evidence"]:
                evidences.append(_load_fever_evidence(evidence, wiki_db))
                labels.append(label)
                claims.append(claim)

    return claims, evidences, labels


def read_fever_dataset(file_path: str, wiki_db):
    if "shared_" in file_path:
        return read_fever_shared(file_path)
    else:
        return read_fever_base(file_path, wiki_db)


def read_fever_dataset_reannotation(file_path: str):
    return read_fever_shared(file_path)


def read_vitaminc_dataset(file_path: str) -> tuple[list, list, list]:
    claims = []
    evidences = []
    labels = []
    with open(file_path) as f:
        for line in f:
            line_loaded = json.loads(line)
            claims.append(line_loaded['claim'])
            evidences.append(line_loaded['evidence'])
            label_mapped = properties.LABEL_DICT[properties.Label(line_loaded["label"].lower())]
            labels.append(label_mapped)

    return claims, evidences, labels


def read_hover_dataset(file_path: str, wiki_db):
    claims = []
    evidences = []
    labels = []
    with open(file_path) as f:
        hover_data = json.load(f)

    for entry in hover_data:
        claims.append(entry['claim'])
        labels.append(properties.LABEL_DICT[properties.Label(entry['label'].lower())])
        evidences.append(_load_hover_evidence(entry['supporting_facts'], wiki_db))

    return claims, evidences, labels


def read_averitec_manual_eval_data(file_path: str) -> Tuple[list, list, list]:
    """
    Reads and processes averitec data from manual evaluation.

    :param file_path: path to the manual eval file after majority voting
    :return: list of claims, evidence and labels
    """
    print("file_path: {}".format(file_path))
    if file_path.endswith(".csv"):
        dataset = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        dataset = pd.read_excel(file_path, header=0)
    else:
        raise ValueError(
            "Exception while reading averitec manual eval data, 'file_path' should either be a .csv or a .xlsx file.")
    claims = []
    labels = []
    evidences = []

    for i, row in dataset.iterrows():
        labels.append(properties.LABEL_DICT[properties.Label(row['label_majority'].strip().lower())])
        claims.append(row['claim'])
        evidences.append(row['predicted evidence'].replace("\n", " ").replace("\n", " "))

    sample_index = 0
    print("claim: {}".format(claims[sample_index]))
    print("predicted evidence: {}".format(evidences[sample_index]))
    print("maj label df: {}".format(dataset.iloc[sample_index]['label_majority']))
    print("maj label: {}".format(labels[sample_index]))
    print("converted label: {}".format(properties.Label(labels[sample_index])))
    print("converted label 2: {}".format(properties.LABEL_DICT[properties.Label(labels[sample_index])]))

    return claims, evidences, labels


def percentage_difference(a: float, b: float) -> float:
    return ((b - a) / a) * 100


def remove_special_characters(text):
    # Define the pattern to keep only alphanumeric characters and spaces
    pattern = r'[^A-Za-z0-9 ]+'
    # Substitute the pattern with an empty string
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


def normalize_text(text):
    return ' '.join(text.split())


def read_averitec_dataset(file_path: str, filter_conflicting_evid: bool = True) -> Tuple[list, list, list]:
    """
    Reads averitec dataset from given path.
    :param file_path: path to .json/.jsonl file containing Averitec data
    :param filter_conflicting_evid:
    :return: list of claims, evidences, and labels
    """
    print("Filepath is: {}".format(file_path))
    if file_path.endswith(".jsonl"):
        dataset = load_jsonl_file(file_path)
    else:
        dataset = load_json_file(file_path)
    claims = []
    qa_pairs = []
    labels = []
    # iterate
    for entry in dataset:
        # if filter_conflicting_evid:
        # if entry["label"] == "Conflicting Evidence/Cherrypicking":
        #     continue

        if "label" in entry:
            labels.append(properties.LABEL_DICT[properties.Label(entry["label"].lower())])
        else:
            try:
                labels.append(properties.LABEL_DICT[properties.Label(entry["pred_label"].lower())])
            except Exception as e:
                print(e)
                continue
        claims.append(entry["claim"])

        qa_pair = ""
        if "questions" in entry:
            evidence_field = "questions"
        else:
            evidence_field = "evidence"
        # for qa in entry["questions"]: todo uncomment later
        for qa in entry[evidence_field]:
            qa_pair += (qa["question"] + " ")
            if "answers" in qa:
                for a in qa["answers"]:
                    qa_pair += (a["answer"] + " ")
                    if "answer_type" in a and a["answer_type"] == "Boolean":
                        qa_pair += ("." + a["boolean_explanation"] + ". ")
            else:
                qa_pair += (qa["answer"] + " ")

        qa_pairs.append(qa_pair)

    return claims, qa_pairs, labels


def read_averitec_before_after_p4(file_path):
    claims = []
    qa_pairs = []
    labels = []
    # load file
    with open(file_path) as f:
        for line in f:
            line_loaded = json.loads(line)
            if line_loaded["label"].lower == "conflicting evidence/cherrypicking":
                # scorer for 3-class problems
                continue
            claims.append(line_loaded['claim'])
            label_mapped = properties.LABEL_DICT[properties.Label(line_loaded["label"].lower())]
            labels.append(label_mapped)
            qa_pair = ""
            for qa in line_loaded["evidence"]:
                qa_pair += (qa["question"] + " ")
                for a in qa["answers"]:
                    qa_pair += (a["answer"] + " ")
                    if a["answer_type"] == "Boolean":
                        qa_pair += ("." + a["boolean_explanation"] + ". ")
            qa_pairs.append(qa_pair)
    return claims, qa_pairs, labels


def to_dict(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__))


def save_jsonl_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(to_dict(entry), f)
            f.write("\n")


def load_jsonl_file(file_path, dataclass=None):
    content = []
    with open(file_path, "r", encoding="utf-8") as f:
        for entry in f.readlines():
            if dataclass:
                try:
                    content.append(dacite.from_dict(data_class=dataclass, data=json.loads(entry)))
                except Exception as e:
                    print("Exception during loading data into dataclass {}: {}".format(type(dataclass), e))
                    continue
            else:
                entry = json.loads(entry)
                if type(entry) == dict:
                    content.append(entry)
                elif type(entry) == list:
                    for e in entry:
                        content.append(e)
    return content


def _extract_averitec_evidence(averitec_questions: list):
    evidence = ""
    for qa in averitec_questions:
        evidence += (qa["question"] + " ")
        for a in qa["answers"]:
            evidence += (a["answer"] + " ")
            if a["answer_type"] == "Boolean":
                evidence += (" " + a["boolean_explanation"] + " ")
    return evidence


def map_averitec_to_dataclass_format(averitec: dict, extract_evidence=False):
    """Formats Averitec dataset files to match fields specified in properties.AveritecEntry."""
    if not extract_evidence:
        return dacite.from_dict(data_class=properties.AveritecEntry,
                                data={"claim": averitec["claim"], "label": averitec["label"],
                                      "justification": averitec["justification"],
                                      "evidence": averitec["questions"]})
    else:
        return dacite.from_dict(data_class=properties.AveritecEntry,
                                data={"claim": averitec["claim"], "label": averitec["label"],
                                      "justification": averitec["justification"],
                                      "evidence": _extract_averitec_evidence(averitec['questions'])})


def load_averitec_base(path: str, extract_evidence=False) -> List[properties.AveritecEntry]:
    """Loads and formats Averitec dataset files (train, test, or dev)."""
    return [map_averitec_to_dataclass_format(entry, extract_evidence) for entry in load_json_file(path)]


def map_fever_to_dataclass_format(fever_entry: list):
    """Formats Averitec dataset files to match fields specified in properties.AveritecEntry."""
    mapped_entries = []
    for fever_subentry in iter(fever_entry[1].values()):
        evidence = ". ".join(e[2].strip() for e in fever_subentry["evidence"] if len(e) > 2).replace("..", ".")
        mapped_entries.append(dacite.from_dict(data_class=properties.AveritecEntry,
                                               data={"claim": fever_subentry["claim"], "label": fever_subentry["label"],
                                                     "justification": "",
                                                     "evidence": evidence}))
    return mapped_entries


def load_fever(path: str) -> List[properties.AveritecEntry]:
    """Loads and formats Fever files."""
    fever_entries = []
    for entry in load_jsonl_file(path):
        fever_entries.extend(map_fever_to_dataclass_format(entry))
    return fever_entries


def calc_meteor(reference: str, candidate: str):
    return metric_meteor.compute(predictions=[candidate], references=[reference])['meteor']


def _pairwise_meteor(reference, candidate):  # Todo this is not thread safe, no idea how to make it so
    return nltk.translate.meteor_score.single_meteor_score(word_tokenize(reference), word_tokenize(candidate))


def _compute_all_pairwise_scores(src_data, tgt_data, metric):
    X = np.empty((len(src_data), len(tgt_data)))

    for i in range(len(src_data)):
        for j in range(len(tgt_data)):
            X[i][j] = (metric(src_data[i], tgt_data[j]))

    return X


def prepare_dataset(claims, evidence, labels, tokenizer):
    data_tokenized = tokenizer(evidence, claims,
                               max_length=1024,
                               truncation=True,
                               padding=True, return_tensors="pt")

    return PseudoTrainedScorerDataset(data_tokenized, labels)


def calc_hungarian_meteor(candidate: str, reference: str, only_question=False):
    """
    Computation of hungarian metrics as per Schlichtkrull et al. (2023)
    :param candidate:
    :param reference:
    :return:
    """
    if only_question:
        src_data = ["".join(x.split("Question:")) for x in reference.split("\n") if "Question" in x]
        tgt_data = ["".join(x.split("Question:")) for x in candidate.split("\n") if "Question" in x]
    else:
        src_data = ["Question" + x.strip() for x in reference.split("Question") if x]
        tgt_data = ["Question" + x.strip() for x in candidate.split("Question") if x]

    pairwise_scores = _compute_all_pairwise_scores(src_data, tgt_data, calc_meteor)
    assignment = scipy.optimize.linear_sum_assignment(pairwise_scores, maximize=True)
    assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()

    # Reweight to account for unmatched target questions
    reweight_term = 1 / float(len(candidate))
    assignment_utility *= reweight_term
    return assignment_utility
    return metric_meteor.compute(predictions=[candidate], references=[reference])['meteor']


def calc_bleu(candidate: str, reference: str):
    return metric_bleu.compute(predictions=[candidate.split()], references=[[reference.split()]])['bleu']


def calc_rouge(candidate: str, reference: str):
    return metric_rouge.compute(predictions=[candidate], references=[reference])['rougeL'].mid.fmeasure


def _load_hover_evidence(evidences: list, wiki_db):
    evidence_text = ""
    for e in evidences:
        try:
            doc = \
                wiki_db.execute("SELECT * FROM documents WHERE id=(?)",
                                (unicodedata.normalize('NFD', e[0]),)).fetchall()[
                    0]
            # retrieve relevant sentence as evidence
            evidence_text += sent_tokenize(doc[1])[e[1] - 1]  # sentence id is 1 or larger (excl. 0)
        except Exception as e:
            print(e)
            continue
    return evidence_text


def _map_hover_to_dataclass_format(hover_entry: dict, wiki_db) -> properties.AveritecEntry:
    """Formats Hover dataset files to match fields specified in properties.AveritecEntry."""
    return dacite.from_dict(data_class=properties.AveritecEntry,
                            data={"claim": hover_entry["claim"], "label": hover_entry["label"],
                                  "evidence": _load_hover_evidence(hover_entry["supporting_facts"], wiki_db)})


def load_hover(path: str, wiki_db) -> List[properties.AveritecEntry]:
    """Loads and formats Hover dataset."""
    entries = []
    for entry in load_json_file(path):
        entries.append(_map_hover_to_dataclass_format(entry, wiki_db))
    return entries


def connect_to_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    return c


def prepare_df_sample(claims, evidences, labels):
    ds_entries = []
    for claim, evid, label in zip(claims, evidences, labels):
        ds_entry = SAMPLE_DICT.copy()
        ds_entry['claim'] = claim
        ds_entry['evidence'] = evid
        ds_entry['label'] = properties.LABEL_DICT_REVERSE[label]
        ds_entries.append(ds_entry)
    return ds_entries


def prepare_and_save(claims, evidences, labels, path):
    data_formatted = prepare_df_sample(claims, evidences, labels)
    save_jsonl_file(data_formatted, path)

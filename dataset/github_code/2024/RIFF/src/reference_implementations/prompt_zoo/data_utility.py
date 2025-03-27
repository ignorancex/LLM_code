"""This module implements the functions for preprocessing the data files into
pytorch dataloaders."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from absl import flags
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5Tokenizer

FLAGS = flags.FLAGS
flags.DEFINE_integer("train_batch_size", 16, "The batch size used for training.")
flags.DEFINE_integer("eval_batch_size", 2048, "The batch size used for inference on the test or validation data.")
flags.DEFINE_integer("source_max_length", 128, "The maximum number of tokens consider in the input sequence.")
flags.DEFINE_integer("decoder_max_length", 128, "The maximum number of tokens consider in the output sequence.")
flags.DEFINE_string(
    "classification_type",
    "fewshot",
    "If fewshot, pick 16 examples per labels, pick the same number of train examples for the development set.",
)
flags.DEFINE_integer("fewshot_sample_size", 16, "size of samples to take for fewshot learning per class.")
flags.DEFINE_integer("seed", 42, "the seed number")
flags.DEFINE_string("instruction_type", "qa", "The instruction type to format the input sentences.")
flags.DEFINE_string("lm_type", "roberta", "which lm type? t5 or roberta?")


@dataclass
class ClassificationRawData:
    """Input/Classes for classification raw data."""

    inputs: List[str]
    gold_outputs: List[str]
    paraphrase_inputs: List[str]


def white_space_fix(text: str) -> str:
    """Remove extra spaces in text."""
    return " ".join(text.split())


def return_class_to_id() -> Dict[str, str]:
    if "_sst2_" in FLAGS.instruction_type:
        return {"terrible": "0", "great": "1"}
    if "_mr_" in FLAGS.instruction_type:
        return {"terrible": "0", "great": "1"}
    if "_cr_" in FLAGS.instruction_type:
        return {"terrible": "0", "great": "1"}
    elif "_sst5_" in FLAGS.instruction_type:
        return {"terrible": "0", "bad": "1", "okay": "2", "good": "3", "great": "4"}
    elif "_agnews_" in FLAGS.instruction_type:
        return {"World": "0", "Sports": "1", "Business": "2", "Tech": "3"}
    elif "_trec_" in FLAGS.instruction_type:
        return {"Description": "0", "Entity": "1", "Expression": "2", "Human": "3", "Location": "4", "Number": "5"}
    elif "_subj_" in FLAGS.instruction_type:
        return {"subjective": "0", "objective": "1"}
    return {}


def return_instruction() -> Tuple[str, str, str]:
    """Return the instruction type."""
    instruction = ""
    if "_sst2_" in FLAGS.instruction_type:
        instruction = "In this task, you are given sentences from movie reviews. \
            The task is to classify a sentence as 'great' if the sentiment of the \
            sentence is positive or as 'terrible' if the sentiment of the sentence is negative."
        template = "<s> {instruction} {sentence} . It was <mask> . </s>"
    elif "_cr_" in FLAGS.instruction_type:
        instruction = "In this task, you are given sentences from customer \
                reviews. The task is to classify a sentence as 'great' if \
                the sentiment of the sentence is positive or as 'terrible' \
                if the sentiment of the sentence is negative."
        template = "<s> {instruction} {sentence} . It was <mask> . </s>"
    elif "_mr_" in FLAGS.instruction_type:
        instruction = "In this task, you are given sentences from movie \
                reviews. The task is to classify a sentence as 'great' if \
                the sentiment of the sentence is positive or as 'terrible' \
                if the sentiment of the sentence is negative."
        template = "<s> {instruction} {sentence} . It was <mask> . </s>"
    elif "_sst5_" in FLAGS.instruction_type:
        instruction = "In this task, you are given sentences from movie reviews. \
            Based on the given review, classify it to one of the five classes: \
                (1) terrible, (2) bad, (3) okay, (4) good, and (5) great."
        template = "<s> {instruction} {sentence} . It was <mask> . </s>"
    elif "_subj_" in FLAGS.instruction_type:
        instruction = "In this task, you are given sentences from reviews. \
            The task is to classify a sentence as 'subjective' if the \
            opinion of the sentence is subjective or as 'objective' \
            if the opinion of the sentence is objective."
        template = "<s> {instruction} {sentence} . This is <mask> . </s>"
    elif "_trec_" in FLAGS.instruction_type:
        instruction = "You are given a question. You need to detect which \
            category better describes the question. Answer with \
            'Description', 'Entity', 'Expression', 'Human', 'Location', and 'Number'."
        template = "<s> {instruction} <mask>: {sentence} . </s>"
    elif "_agnews_" in FLAGS.instruction_type:
        instruction = "In this task, you are given a news article. Your task is to classify \
            the article to one out of the four topics 'World', 'Sports', 'Business', 'Tech' \
            if the article's main topic is relevant to the world, sports, business, \
            and technology, correspondingly. If you are not sure about the topic, choose the closest option."
        template = "<s> {instruction} <mask> News: {sentence} . </s>"

    t5_template = template.replace("<s> ", "").replace("<mask>", "<extra_id_0>")
    if FLAGS.lm_type in ["roberta", "llama2"]:
        return white_space_fix(instruction), white_space_fix(template), "{label}"
    elif FLAGS.lm_type == "t5":
        return white_space_fix(instruction), white_space_fix(t5_template), "{label} </s>"
    return "", "", ""


def tokenize_samples(batch: torch.utils.data.Dataset, samples: List[str], tokenizer: AutoTokenizer) -> None:
    samples = [white_space_fix(f"{sample} </s>") for sample in samples]
    output_encodings = tokenizer(
        samples,
        truncation=True,
        padding="max_length",
        max_length=FLAGS.source_max_length,
        add_special_tokens=False,
    )
    batch["para_target_attention_mask"] = torch.tensor(output_encodings.attention_mask)
    batch["para_labels"] = torch.tensor(output_encodings.input_ids)


def augment_batch(
    batch: torch.utils.data.Dataset,
    paraphrases: List[str],
    tokenizer: AutoTokenizer,
    num_return_seq: int,
    remove_instruction: Optional[bool] = False,
) -> None:
    """Augment the batch with paraphrases."""
    batch_size, seq_len = batch["input_ids"].size()
    input_ids = batch.pop("input_ids").reshape(batch_size, 1, seq_len)
    attention_mask = batch.pop("attention_mask").reshape(batch_size, 1, seq_len)

    inputs = []
    instruction, template, label_template = return_instruction()
    for index in range(batch_size):
        for par_index in range(num_return_seq):
            par_base_index = index * num_return_seq
            paraphrase = paraphrases[par_base_index + par_index : par_base_index + par_index + 1][0]
            input_str = template.format(instruction=instruction, sentence=paraphrase.removesuffix("."))
            if remove_instruction:
                # for gradient_search.
                input_str = input_str.replace(f" {instruction} ", " ")
            inputs.append(input_str)

    input_encodings = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=FLAGS.source_max_length,
        add_special_tokens=False,
    )

    par_input_ids = torch.tensor(input_encodings.input_ids).reshape(batch_size, num_return_seq, seq_len)
    par_attention_mask = torch.tensor(input_encodings.attention_mask).reshape(batch_size, num_return_seq, seq_len)

    batch["input_ids"] = torch.cat((input_ids, par_input_ids), dim=1).reshape(-1, seq_len)
    batch["attention_mask"] = torch.cat((attention_mask, par_attention_mask), dim=1).reshape(-1, seq_len)


def template_data(sentences: List[str], labels: List[str], id_to_class: Dict[str, str]) -> ClassificationRawData:
    """Helper function to format the data for the models.

    if with_instructions is True, we will add an instruction to the
    input sentence and make the input a template with special keywords
    "instructions:", "sentence:", and "sentiment:".

    if the repeat_input is True, we will repeat the input multiple times
    for every possible output class.

    Finally, the end of sentence token </s> used with T5 models are
    added to both input and output.
    """
    instruction, template, label_template = return_instruction()
    input_sentences = [template.format(instruction=instruction, sentence=sent.removesuffix(".")) for sent in sentences]
    if "_no_instruction" in FLAGS.instruction_type:
        input_sentences = [in_sent.replace(instruction, "") for in_sent in input_sentences]
    input_sentences = [white_space_fix(in_sent) for in_sent in input_sentences]
    gold_outputs = [label_template.format(label=label) for label in labels]
    paraphrase_inputs = [white_space_fix(f"{sent} </s>") for sent in sentences]
    return ClassificationRawData(
        inputs=input_sentences,
        gold_outputs=gold_outputs,
        paraphrase_inputs=paraphrase_inputs,
    )


def read_fewshot_file(class_to_id: Dict[str, str], file_path: str) -> ClassificationRawData:
    """Load the fewshot files."""
    df = pd.read_csv(file_path, sep="\t")
    if "text" in df:
        sentences = df.text.tolist()
    else:
        sentences = df.sentence.tolist()

    id_to_class = {id: label for label, id in class_to_id.items()}

    labels = [id_to_class[str(id)] for id in df.label.tolist()]
    return template_data(sentences, labels, id_to_class)


class ClassificationDataset(Dataset):
    """Subclass the pytorch's Dataset to build my own dataset for the text classification
    task."""

    def __init__(self, data: Dict[str, Union[List[int], List[List[int]]]]) -> None:
        """Store the reference to the tokenized data."""
        self.data = data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return the elements for example index 'idx' as a dictionary with
        tensor values."""
        ret = {}
        for key, val in self.data.items():
            if isinstance(val[idx], str):
                ret[key] = val[idx]
            else:
                ret[key] = torch.tensor(val[idx])
        return ret

    def __len__(self) -> int:
        """Return the length of the data."""
        return len(self.data["input_ids"])


def tokenize_data(
    rawdata: ClassificationRawData, tokenizer: T5Tokenizer, para_tokenizer: Optional[T5Tokenizer] = None
) -> ClassificationDataset:
    """Tokenize data into a dataset."""
    input_encodings = tokenizer(
        rawdata.inputs,
        truncation=True,
        padding="max_length",
        max_length=FLAGS.source_max_length,
        add_special_tokens=False,
    )

    data = {
        "input_ids": input_encodings.input_ids,
        "attention_mask": input_encodings.attention_mask,
        "gold_outputs": rawdata.gold_outputs,
        "paraphrase_inputs": rawdata.paraphrase_inputs,
    }

    if para_tokenizer is not None:
        para_input_encodings = para_tokenizer(
            rawdata.paraphrase_inputs,
            truncation=True,
            padding="max_length",
            max_length=FLAGS.source_max_length,
            add_special_tokens=False,
        )
        data["para_input_ids"] = para_input_encodings.input_ids
        data["para_attention_mask"] = para_input_encodings.attention_mask

    return ClassificationDataset(data)


def create_sentiment_dataset(
    tokenizer: AutoTokenizer,
    train_file_name: Optional[str] = None,
    dev_file_name: Optional[str] = None,
    test_file_name: Optional[str] = None,
    task_name: Optional[str] = None,
    para_tokenizer: Optional[T5Tokenizer] = None,
) -> DataLoader:
    """Function to create the required hugging-face dataset to train the LM
    models."""

    # create a class to id in the tokenizer.
    class_to_id = return_class_to_id()
    tokenizer.class_to_id = class_to_id
    tokenizer.id_to_class = {id: label for label, id in class_to_id.items()}

    if task_name not in ["sst2", "sst5", "mr", "cr", "trec", "subj", "agnews"]:
        raise ValueError(f"this {task_name} is not supported!")

    if train_file_name is not None:
        rawdata = read_fewshot_file(class_to_id, train_file_name)
        shuffle = True

    if dev_file_name is not None:
        rawdata = read_fewshot_file(class_to_id, dev_file_name)
        shuffle = False

    if test_file_name is not None:
        rawdata = read_fewshot_file(class_to_id, test_file_name)
        shuffle = False

    dataset = tokenize_data(rawdata, tokenizer, para_tokenizer)
    dataloader = DataLoader(dataset, batch_size=FLAGS.train_batch_size, shuffle=shuffle)
    return dataloader

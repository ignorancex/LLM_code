import dotenv

dotenv.load_dotenv()

import json
import os
from functools import partial
from pprint import pprint

import click
import datasets
import numpy as np
from omegaconf import OmegaConf
from utils import jprint


def load_pubmedqa(seed=42, get_custom_test_split=False):
    """
    https://huggingface.co/datasets/qiaojin/PubMedQA

    Data split, 500 train, 500 test can be found in https://github.com/pubmedqa/pubmedqa/blob/master/preprocess/split_dataset.py

    Format is from https://github.com/FreedomIntelligence/HuatuoGPT-o1/blob/main/evaluation/data/eval_data.json
    """

    path = "qiaojin/PubMedQA"
    name = "pqa_labeled"
    split = "train"
    dataset = datasets.load_dataset(path, name=name, split=split)

    def _map_pubmedqa(x, source_name, rng):
        context = "\n".join(x["context"]["contexts"])
        question = x["question"]
        final_decision = x["final_decision"]
        long_answer = x["long_answer"]

        # NOTE: build options
        options = ["yes", "no", "maybe"]
        if final_decision not in options:
            raise ValueError(f"final_decision {final_decision} not in {options}")

        # NOTE: shuffle options to avoid memorization
        rng.shuffle(options)
        answer_idx = options.index(final_decision)
        answer_letter = chr(ord("A") + answer_idx)
        answer_string = final_decision

        letter_options = {chr(ord("A") + i): ans for i, ans in enumerate(options)}

        # NOTE: options format
        options = "\n".join([f"{op}. {ans}" for op, ans in letter_options.items()])

        # NOTE: prompt template, different datasets may vary
        prompt_template = "{context}\n{question}\n{options}"
        prompt = prompt_template.format(
            context=context, question=question, options=options
        )

        return {
            # related
            # "context": context,
            # "question": question,
            # "final_decision": final_decision,
            # "long_answer": long_answer,
            # source
            "source": source_name,
            "metadata": str(x),
            # llm inputs
            "prompt": prompt,
            "answer_letter": answer_letter,
            "answer_idx": answer_idx,
            "answer_string": answer_string,
        }

    print("Dataset info:")
    jprint(dataset[0])

    rng = np.random.default_rng(seed=seed)
    source_name = f"{path}:{name}"
    mapped_sample = _map_pubmedqa(dataset[0], source_name=source_name, rng=rng)
    print("\nMapped sample:")
    jprint(mapped_sample)

    mapped_dataset: datasets.Dataset
    mapped_dataset = dataset.map(
        partial(_map_pubmedqa, source_name=source_name, rng=rng),
        remove_columns=dataset.column_names,
    )

    # NOTE: split dataset, 500: 500
    # https://huggingface.co/docs/datasets/v3.3.2/en/process#split
    # seed: 0 https://github.com/pubmedqa/pubmedqa/blob/master/preprocess/split_dataset.py
    PUBMEDQA_SEED = 0
    mapped_dataset = mapped_dataset.train_test_split(
        test_size=0.5, seed=PUBMEDQA_SEED, shuffle=True
    )

    if get_custom_test_split:
        mapped_dataset = mapped_dataset["test"]
    else:
        mapped_dataset = mapped_dataset["train"]
    return mapped_dataset


@click.command()
@click.option("--debug", is_flag=True)
@click.option("--upload_test", is_flag=True)
def main(debug=False, upload_test=False):
    dataset = load_pubmedqa()

    if upload_test:
        test_dataset = load_pubmedqa(get_custom_test_split=True)
        test_dataset.push_to_hub("mmqm/pubmedqa_custom_test")
    if debug:
        # fmt: off
        import IPython; IPython.embed()
        # fmt: on


if __name__ == "__main__":
    main()

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


def load_medqa(seed=42):
    """
    https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options
    """
    name = "GBaker/MedQA-USMLE-4-options"
    split = "train"
    dataset = datasets.load_dataset(name, split=split)

    def _map_medqa(x, source_name, rng):
        raw_options = x["options"]
        question = x["question"]
        answer = x["answer_idx"]
        source = source_name

        answer_string = raw_options[answer]
        options = list(raw_options.values())
        rng.shuffle(options)
        answer_idx = options.index(answer_string)
        answer_letter = chr(ord("A") + answer_idx)
        letter_options = {chr(ord("A") + i): ans for i, ans in enumerate(options)}

        # NOTE: options format
        options = "\n".join([f"{op}. {ans}" for op, ans in letter_options.items()])

        # NOTE: prompt template, different datasets may vary
        prompt_template = "{question}\n{options}"
        prompt = prompt_template.format(question=question, options=options)

        return {
            # related
            # "question": question,
            # "answer": answer,
            # "options": raw_options,
            # source
            "source": source,
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
    source_name = f"{name}"
    mapped_sample = _map_medqa(dataset[0], source_name=source_name, rng=rng)

    print("\nMapped sample:")
    jprint(mapped_sample)

    mapped_dataset: datasets.Dataset
    mapped_dataset = dataset.map(
        partial(_map_medqa, source_name=source_name, rng=rng),
        remove_columns=dataset.column_names,
    )

    return mapped_dataset


@click.command()
@click.option("--debug", is_flag=True)
def main(debug=False):
    dataset = load_medqa()

    if debug:
        # fmt: off
        import IPython; IPython.embed()
        # fmt: on


if __name__ == "__main__":
    main()

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


def load_headqa(seed=42):
    """
    https://huggingface.co/datasets/openlifescienceai/headqa
    """
    name = "openlifescienceai/headqa"
    split = "train"
    dataset = datasets.load_dataset(name, split=split)

    def _map_headqa(x, source_name, rng):
        """
        {
        "Correct Answer": "Internal mitochondrial",
        "Correct Option": "A",
        "Options": {
        "A": "Internal mitochondrial",
        "B": "External mitochondrial",
        "C": "Plasma.",
        "D": "Lysosomal"
        },
        "Question": "The cardiolipin phospholipid is abundant in the membrane:"
        }
        """
        data = x["data"]

        question = data["Question"]
        raw_answer_string = data["Correct Answer"]
        raw_answer_idx = data["Correct Option"]
        raw_options = data["Options"]
        if raw_answer_string != raw_options[raw_answer_idx]:
            raise ValueError(
                f"Answer not in options, `{raw_answer_string}` not in `{raw_options}`"
            )

        options = list(raw_options.values())
        rng.shuffle(options)
        answer_idx = options.index(raw_answer_string)
        answer_letter = chr(ord("A") + answer_idx)
        letter_options = {chr(ord("A") + i): ans for i, ans in enumerate(options)}
        answer_string = raw_answer_string

        # NOTE: options format
        options = "\n".join([f"{op}. {ans}" for op, ans in letter_options.items()])

        # NOTE: prompt template, different datasets may vary
        prompt_template = "{question}\n{options}"
        prompt = prompt_template.format(question=question, options=options)
        return {
            # related
            # "question": question,
            # "raw_answer_string": raw_answer_string,
            # "raw_answer_idx": raw_answer_idx,
            # "options": raw_options,
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
    source_name = f"{name}"
    mapped_sample = _map_headqa(dataset[0], source_name=source_name, rng=rng)

    print("\nMapped sample:")
    jprint(mapped_sample)

    mapped_dataset: datasets.Dataset
    mapped_dataset = dataset.map(
        partial(_map_headqa, source_name=source_name, rng=rng),
        remove_columns=dataset.column_names,
    )

    return mapped_dataset


@click.command()
@click.option("--debug", is_flag=True)
def main(debug=False):
    dataset = load_headqa()

    if debug:
        # fmt: off
        import IPython; IPython.embed()
        # fmt: on


if __name__ == "__main__":
    main()

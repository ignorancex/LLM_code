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


def load_medmcqa(seed=42):
    """
    https://huggingface.co/datasets/openlifescienceai/medmcqa
    """
    path = "openlifescienceai/medmcqa"
    split = "train"
    dataset = datasets.load_dataset(path, split=split)

    medmcqa_answer_mapping = dict(enumerate("ABCD"))

    def _map_medmcqa(x, source_name, answer_mapping, rng):
        question = x["question"]

        raw_options = {
            "A": x["opa"],
            "B": x["opb"],
            "C": x["opc"],
            "D": x["opd"],
        }
        raw_answer_idx = x["cop"]
        answer_letter = answer_mapping[raw_answer_idx]
        answer_string = raw_options[answer_letter]

        options = list(raw_options.values())
        rng.shuffle(options)
        answer_idx = options.index(answer_string)
        answer_letter = chr(ord("A") + answer_idx)
        letter_options = {chr(ord("A") + i): ans for i, ans in enumerate(options)}

        options = "\n".join([f"{op}. {ans}" for op, ans in letter_options.items()])

        # NOTE: prompt template, different datasets may vary
        prompt_template = "{question}\n{options}"
        prompt = prompt_template.format(question=question, options=options)

        return {
            # related
            # "question": question,
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
    source_name = path
    mapped_sample = _map_medmcqa(
        dataset[0],
        source_name=source_name,
        answer_mapping=medmcqa_answer_mapping,
        rng=rng,
    )
    print("\nMapped sample:")
    jprint(mapped_sample)

    mapped_dataset: datasets.Dataset
    mapped_dataset = dataset.map(
        partial(
            _map_medmcqa,
            source_name=source_name,
            rng=rng,
            answer_mapping=medmcqa_answer_mapping,
        ),
        remove_columns=dataset.column_names,
    )

    return mapped_dataset


@click.command()
@click.option("--debug", is_flag=True)
def main(debug=False):
    dataset = load_medmcqa()

    if debug:
        # fmt: off
        import IPython; IPython.embed()
        # fmt: on


if __name__ == "__main__":
    main()

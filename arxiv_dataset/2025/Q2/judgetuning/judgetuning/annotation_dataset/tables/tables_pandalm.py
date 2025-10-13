import random
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from judgetuning.annotation_dataset.tables.table_annotation_dataset import (
    TableAnnotationDataset,
    table_root_default,
)
from judgetuning.utils import random_string

dataset_name = "pandalm"


def generate_instructions(df, table_root, name):
    # Note: updates df with the concatenated "instruction" and "input" fields
    def create_index(instruction: str) -> str:
        x = (
            instruction[:20].replace(" ", "-").lower()
            + "...-"
            + random_string(length=5)
        )
        return x.encode("utf-8", errors="replace").decode("utf-8")

    df_instruction = df.copy()
    # this dataset has an "instruction" and an "input" column. The first is to be applied to the latter
    # when present.
    df_instruction["instruction"] = df_instruction.apply(
        lambda row: (
            row["instruction"] + "\n" + row["input"]
            if row["input"]
            else row["instruction"]
        ),
        axis=1,
    )

    # Ugly whacky, update df with the concatenated "instruction" and "input" fields
    df["instruction"] = df_instruction["instruction"]

    df_instruction["domain"] = name
    # df_instruction['instruction'] = df_instruction['instruction'].map(
    #     lambda x: x.encode('unicode-escape').decode('utf-8'))
    df_instruction.drop_duplicates(subset="instruction", inplace=True)
    df_instruction["instruction_index"] = df_instruction["instruction"].apply(
        create_index
    )
    df_instruction.to_csv(
        table_root / "instructions" / f"{name}.csv", index=False, errors="replace"
    )
    instructions, instructions_index = df_instruction.loc[
        :, ["instruction", "instruction_index"]
    ].values.T
    return dict(zip(instructions, instructions_index))


def generate_model_outputs(df, table_root, name, instruction_index_dict):
    df_model_rows = []
    for i, row in df.iterrows():
        model1, model2 = row["cmp_key"].split("_")
        instruction_index = instruction_index_dict[row["instruction"]]
        df_model_rows.append(
            {
                "model": model1,
                "instruction_index": instruction_index,
                "output": row["response1"],
            }
        )
        df_model_rows.append(
            {
                "model": model2,
                "instruction_index": instruction_index,
                "output": row["response2"],
            }
        )
    df_model = pd.DataFrame(df_model_rows)
    df_model = df_model.drop_duplicates(["instruction_index", "model"])
    df_model.to_csv(
        table_root / "model_outputs" / f"{name}.csv.zip", index=False, errors="replace"
    )


def generate_annotations(df, table_root, name, instruction_index_dict):
    df_judge_rows = []
    for i, row in df.iterrows():
        model1, model2 = row["cmp_key"].split("_")
        common_kwargs = {
            "model1": model1,
            "model2": model2,
            "output1": str(row["response1"]),
            "output2": str(row["response2"]),
            "instruction_index": instruction_index_dict[row["instruction"]],
            "prompt": "NA",
            "swap": False,
            "number_output_tokens": 0,
            # TODO we could put avg salary?
            "cost": 0,
            "time": 0,
        }
        preferences = []
        for j in range(1, 4):
            col = f"annotator{j}"
            # can be 0, 1, 2 which corr. to tie, win model 1 and win model 2
            raw_pref = row[col]
            match raw_pref:
                case 0:
                    preference = 0.5
                case 1:
                    preference = 0.0
                case 2:
                    preference = 1.0
                case _:
                    raise ValueError(f"Invalid preference {raw_pref}")
            df_judge_rows.append(
                {
                    "preference": preference,
                    "judge_index": col,
                    **common_kwargs,
                }
            )
            preferences.append(preference)
        df_judge_rows.append(
            {
                "preference": np.mean(preferences),
                "judge_index": "annotator-average",
                **common_kwargs,
            }
        )
        # compute the mode of annotators for each instructions
        unique, counts = np.unique(preferences, return_counts=True)
        mode_index = np.argmax(counts)
        mode_value = unique[mode_index]
        df_judge_rows.append(
            {
                "preference": mode_value,
                "judge_index": "annotator-mode",
                **common_kwargs,
            }
        )
        
    df_judge = pd.DataFrame(df_judge_rows)
    df_judge.to_parquet(
        table_root / "judge_annotations" / f"{name}.parquet", index=False
    )


def generate_dataset():
    random.seed(0)
    np.random.seed(0)

    table_root = table_root_default
    pandalm_root = table_root / dataset_name
    pandalm_root.mkdir(exist_ok=True)

    # download pandalm file if does not exists
    local_file = pandalm_root / "testset-v1.json"
    if not local_file.exists():
        print(f"Downloading {local_file} as does not exists")
        url = "https://raw.githubusercontent.com/WeOpenML/PandaLM/refs/heads/main/data/testset-v1.json"
        response = requests.get(url)
        with open(local_file, "wb") as file:
            file.write(response.content)
    else:
        print(
            f"Reusing existing dataset at {local_file}, remove the file to download again."
        )

    # goes through the json and convert to annotation format
    # this requires generating 1) instruction 2) model 3) annotation files
    df = pd.read_json(local_file)

    # 1) save instructions
    instruction_index_dict = generate_instructions(df, table_root, dataset_name)

    # 2) save model outputs
    generate_model_outputs(df, table_root, dataset_name, instruction_index_dict)

    # 3) save human annotations
    generate_annotations(df, table_root, dataset_name, instruction_index_dict)
    print(f"done generating {dataset_name}")


class PandaLMDataset(TableAnnotationDataset):
    def __init__(
        self,
        table_root: Path = table_root_default,
    ):
        super().__init__(
            name=dataset_name,
            table_root=table_root,
            instruction_filenames=[f"{dataset_name}.csv"],
            model_filenames=[f"{dataset_name}.csv.zip"],
            judge_filenames=[f"{dataset_name}.parquet"],
        )
        print(self)

    @staticmethod
    def generate():
        generate_dataset()


if __name__ == "__main__":
    generate_dataset()
    ds = PandaLMDataset()
    ds.push_to_hf()

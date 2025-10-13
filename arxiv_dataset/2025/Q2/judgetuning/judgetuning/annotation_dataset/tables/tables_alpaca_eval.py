import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from timeblock import Timeblock

from judgetuning.annotation_dataset.tables import default_table_path
from judgetuning.annotation_dataset.tables.table_annotation_dataset import (
    TableAnnotationDataset,
)
from judgetuning.join import load_model_mapping
from judgetuning.utils import random_string

table_root_default = default_table_path()


def _alpaca_eval_path():
    if "ALPACAEVAL_PATH" in os.environ:
        print(f'Loading alpaca-eval from env variable: {os.getenv("ALPACAEVAL_PATH")}')
        return Path(os.getenv("ALPACAEVAL_PATH")).expanduser()
    else:
        return Path(__file__).parent.parent.parent.parent / "alpaca_eval"


def import_instructions_and_model_outputs(
    table_root: Path = table_root_default,
):
    # loop over all "model_outputs.json"
    alpaca_root = _alpaca_eval_path() / "results"

    dfs_model = []
    instructions = {}
    for model_output_path in alpaca_root.rglob("*model_outputs.json"):
        with open(model_output_path, "r", encoding="utf-8") as f:
            model = model_output_path.parent.name
            try:
                model_outputs = json.load(f)
            except Exception as e:
                print(model, str(e))
                raise e
            df = pd.DataFrame(model_outputs)
            skip = False
            for col in ["dataset", "generator", "instruction"]:
                if col not in df.columns:
                    skip = True
            if skip:
                continue
            instructions.update(df.set_index("instruction").loc[:, "dataset"].to_dict())
            df = df.drop("dataset", axis=1).drop("generator", axis=1)
            df["model"] = model
            dfs_model.append(df.loc[:, ["instruction", "model", "output"]])

    df_instruction = pd.DataFrame(
        list(instructions.items()), columns=["instruction", "domain"]
    )
    df_instruction["dataset"] = "alpaca-eval"

    random.seed(0)
    instructions = df_instruction.loc[:, "instruction"]
    instruction_index_dict = dict(
        zip(
            instructions,
            instructions.apply(
                lambda x: x[:20].replace(" ", "-").lower()
                + "...-"
                + random_string(length=5)
            ),
        )
    )

    df_model = pd.concat(dfs_model, ignore_index=True)
    df_model.loc[:, "instruction_index"] = df_model.loc[:, "instruction"].apply(
        lambda x: instruction_index_dict[x]
    )
    df_model.drop("instruction", axis=1, inplace=True)
    # todo make table instruction_index, model, model_output for model
    # todo generate index for instructions
    df_instruction["instruction_index"] = df_instruction["instruction"].apply(
        lambda x: instruction_index_dict[x]
    )
    df_instruction.to_csv(table_root / "instructions" / "alpaca-eval.csv", index=False)
    df_model.to_csv(table_root / "model_outputs" / "alpaca-eval.csv.zip", index=False)
    return df_instruction, df_model


def import_alpaca_eval_annotations(
    df_instruction,
    table_root: Path = table_root_default,
):
    alpaca_eval_path = _alpaca_eval_path()
    dfs = []
    for model_annotations_path in (alpaca_eval_path / "results").rglob(
        "*annotations.json"
    ):
        model = model_annotations_path.parent.parent.name
        judge = model_annotations_path.parent.name
        with open(model_annotations_path, "r") as f:
            df_annotations = json.load(f)
        # check that filenames matches the loaded annotations
        dfs.append(pd.DataFrame(df_annotations))
    df = pd.concat(dfs, ignore_index=True)

    df = df.loc[
        (df.generator_1 == "gpt4_1106_preview")
        & (df.annotator == "weighted_alpaca_eval_gpt4_turbo"),
        :,
    ]
    df = df.rename(
        columns={
            "generator_1": "model1",
            "generator_2": "model2",
            "price_per_example": "cost",
            "time_per_example": "time",
            "raw_completion": "completion",
            "annotator": "judge_index",
        }
    )
    df["preference"] = df.preference - 1
    instruction_to_index_dict = dict(
        df_instruction.loc[:, ["instruction", "instruction_index"]].values
    )
    df["instruction_index"] = df["instruction"].apply(
        lambda str: instruction_to_index_dict[str]
    )  # TODO, pass instruction index
    df["prompt"] = np.nan
    df["number_output_tokens"] = np.nan
    df["swap"] = np.nan
    cols = [
        "judge_index",
        "instruction_index",
        "model1",
        "model2",
        "prompt",
        "completion",
        "number_output_tokens",
        "preference",
        "swap",
        "cost",
        "time",
    ]
    for col in cols:
        assert col in df, col
    df.loc[:, cols].to_parquet(
        table_root / "judge_annotations" / "alpaca-eval.parquet", index=False
    )


def import_alpaca_eval():
    df_instruction, df_model = import_instructions_and_model_outputs()
    import_alpaca_eval_annotations(df_instruction)

    # save to HF


class AlpacaEvalDataset(TableAnnotationDataset):
    def __init__(
        self,
        table_root: Path = table_root_default,
        keep_only_chatbot_arena: bool = True,
        rename_chatbot_arena: bool = True,
    ):
        super().__init__(
            name="alpaca-eval",
            table_root=table_root,
            instruction_filenames=["alpaca-eval.csv"],
            model_filenames=["alpaca-eval.csv.zip"],
            judge_filenames=["alpaca-eval.parquet"],
        )
        if keep_only_chatbot_arena or rename_chatbot_arena:
            alpaca_eval_to_chatbot_arena = dict(
                load_model_mapping("alpaca-eval")
                .loc[:, ["alpaca-eval", "chatbot-arena"]]
                .values
            )
            if keep_only_chatbot_arena and not rename_chatbot_arena:
                self.subselect_models(alpaca_eval_to_chatbot_arena.keys())
            elif rename_chatbot_arena:
                self.rename_models(
                    alpaca_eval_to_chatbot_arena, drop_missing=keep_only_chatbot_arena
                )
        print(f"Loaded: {self}")

    @staticmethod
    def generate():
        import_alpaca_eval()


if __name__ == "__main__":
    cls = AlpacaEvalDataset
    # with Timeblock("Generating alpaca eval") as t:
    #     cls.generate()
    # # todo import script
    model = "claude-2.0"
    with Timeblock("Loading alpaca eval") as t:
        ds = AlpacaEvalDataset(keep_only_chatbot_arena=True, rename_chatbot_arena=True)

    ds.model_output(model)

    print(ds.models)
    ds.instructions[:1]
    print(ds.get_instruction_index(ds.instructions[0]))
    m = ds.model_output("gpt-4-1106-preview")
    for i in range(3):
        m[ds.get_instruction_index(ds.instructions[i])]

    print(ds.judges)

    print(ds.df_winrate_against_baseline(models=[model]))
    print(ds.df_winrate_against_baseline(models=[model]).mean())

    print(ds.num_annotations())
    print(ds.cost_per_annotation(models=[model]))
    print(ds.time_per_annotation(models=[model]))

    print(ds.elo_ratings())

    print(ds.chatbot_arena_elo_ratings())

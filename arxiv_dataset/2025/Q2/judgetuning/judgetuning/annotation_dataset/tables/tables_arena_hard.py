import json
import os
import string
from pathlib import Path
import random

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


def _arena_hard_path():
    if "ARENAHARD_PATH" in os.environ:
        # print(f'Loading arena hard from env variable: {os.getenv("ARENAHARD_PATH")}')
        return Path(os.getenv("ARENAHARD_PATH")).expanduser()
    else:
        return (
            Path(__file__).parent.parent.parent.parent
            / "arena-hard"
            / "data"
            / "arena-hard-v0.1"
        )


def import_instructions_and_model_outputs(
    table_root: Path = table_root_default,
):
    answer_path = _arena_hard_path() / "model_answer"
    assert answer_path.exists(), f"{answer_path} does not exists."
    df_instruction = pd.read_json(_arena_hard_path() / "question.jsonl", lines=True)
    df_instruction.rename(columns={"cluster": "domain"}, inplace=True)

    # TODO we could also use "category" which is "arena-hard-v0.1" as of today
    df_instruction["dataset"] = "arena-hard"
    df_instruction["instruction"] = df_instruction.loc[:, "turns"].apply(
        lambda row: row[0]["content"]
    )
    # Note, we could also use AH original index, we chose not to as it is handful to have the beginning of the question
    # in the index to get an idea about the instruction without joining
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

    df_instruction["instruction_index"] = df_instruction["instruction"].apply(
        lambda x: instruction_index_dict[x]
    )

    question_id_to_index_dict = dict(
        zip(
            df_instruction["question_id"],
            df_instruction["instruction_index"],
        )
    )

    df_models = []
    for path in answer_path.glob("*.jsonl"):
        with open(path, "r") as f:
            # model = path.stem
            df_model = pd.read_json(path, lines=True)

            df_model = df_model.loc[:, ["question_id", "choices", "model_id"]]
            df_model["output"] = df_model.loc[:, "choices"].apply(
                lambda row: (
                    row[0]["turns"][0]["content"] if isinstance(row, list) else ""
                )
            )
            df_model.rename(columns={"model_id": "model"}, inplace=True)
            df_model["instruction_index"] = df_model["question_id"].apply(
                lambda x: question_id_to_index_dict[x]
            )
            df_models.append(df_model.loc[:, ["model", "instruction_index", "output"]])
    df_model = pd.concat(df_models, ignore_index=True)

    df_instruction.to_csv(table_root / "instructions" / "arena-hard.csv", index=False)
    df_model.to_csv(table_root / "model_outputs" / "arena-hard.csv.zip", index=False)

    return question_id_to_index_dict


def extract_preference_from_judgement_string(s: str, alpha: float = 0.25) -> float:
    assert 0 <= alpha < 0.5
    mapping = {
        "A>>B": 0,
        "B<<A": 0,
        "A>B": alpha,
        "B<A": alpha,
        "B>>A": 1,
        "A<<B": 1,
        "B>A": 1 - alpha,
        "A<B": 1 - alpha,
        "A=B": 0.5,
    }
    # we assign a tie in case the score was not properly formatted, this could silent errors though
    return mapping.get(s, 0.5)


def import_annotations(
    question_id_to_index_dict,
    table_root: Path = table_root_default,
):
    res = {}
    judgement_path = _arena_hard_path() / "model_judgment"
    rows = []
    for judge_model_file in judgement_path.rglob("*.jsonl"):
        model = judge_model_file.stem
        # arena-hard naming has upper case for name files and lowercase for internal model name
        model = model.lower()
        df = pd.read_json(judge_model_file, lines=True)
        df.model = df.model.str.lower()
        # llama-2-70b-chat-hf is encoded with llama-2-70b-chat-hf only for llama3 judge
        # and llama-2-70b-chat for other judges
        if "llama-2" in model:
            df.model = df.model.apply(lambda s: s.replace("-hf", ""))
        # Index(['question_id', 'model', 'judge', 'games'], dtype='object')
        for (
            question_id,
            model,
            judge,
            (judgement_without_swap, judgement_with_swap),
        ) in df.values:
            if question_id not in question_id_to_index_dict:
                # print(f"File {judge_model_file} contains an instruction that is unknown: {question_id}, skipping it.")
                continue
            instruction_index = question_id_to_index_dict[question_id]
            for i, judgement in enumerate(
                [judgement_without_swap, judgement_with_swap]
            ):
                swap = bool(i)
                preference = extract_preference_from_judgement_string(
                    judgement["score"]
                )
                if swap:
                    preference = 1 - preference
                row = {
                    "judge_index": judge,
                    "instruction_index": instruction_index,
                    # TODO we could double check by making sure that the output in prompt comes from the right baseline
                    "model1": "gpt-4-1106-preview",
                    "model2": model,
                    "prompt": judgement["user_prompt"],
                    "completion": judgement["judgment"],
                    "number_output_tokens": np.nan,
                    "preference": preference,
                    "swap": swap,
                    "cost": np.nan,
                    "time": np.nan,
                }
                rows.append(row)
    df_judge = pd.DataFrame(rows)
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
        assert col in df_judge, col
    df_judge.loc[:, cols].to_parquet(
        table_root / "judge_annotations" / "arena-hard.parquet", index=False
    )


def import_arena_hard():
    question_id_to_index_dict = import_instructions_and_model_outputs()
    import_annotations(question_id_to_index_dict)


class ArenaHardDataset(TableAnnotationDataset):
    def __init__(
        self,
        load_annotations: bool = True,
        table_root: Path = table_root_default,
        keep_only_chatbot_arena: bool = True,
        rename_chatbot_arena: bool = True,
    ):
        super().__init__(
            name="arena-hard",
            table_root=table_root,
            instruction_filenames=["arena-hard.csv"],
            model_filenames=["arena-hard.csv.zip"],
            judge_filenames=["arena-hard.parquet"] if load_annotations else [],
        )
        if keep_only_chatbot_arena or rename_chatbot_arena:
            arena_hard_to_chatbot_arena = dict(
                load_model_mapping("arena-hard")
                .loc[:, ["arena-hard", "chatbot-arena"]]
                .values
            )
            if keep_only_chatbot_arena and not rename_chatbot_arena:
                self.subselect_models(arena_hard_to_chatbot_arena.keys())
            elif rename_chatbot_arena:
                self.rename_models(
                    arena_hard_to_chatbot_arena, drop_missing=keep_only_chatbot_arena
                )
        print(f"Loaded: {self}")

    @property
    def judges(self) -> list[str]:
        # return list manually to be sure that gpt4 is the first judge since we use the first judge for the default
        return [
            "gpt-4-1106-preview",
            "claude-3-opus-20240229",
            "llama-3-70b-instruct",
            "gemini-1.5-pro-api-preview",
            "claude-3-5-sonnet-20240620",
        ]

    @staticmethod
    def generate():
        import_arena_hard()


if __name__ == "__main__":
    # with Timeblock("Generating arena hard") as t:
    #     ArenaHardDataset.generate()

    cls = ArenaHardDataset
    # with Timeblock("Generating alpaca eval") as t:
    #     cls.generate()
    # # todo import script
    # model = "gemma-2b-it"
    with Timeblock("Loading arena-hard") as t:
        ds = cls()

    model = ds.models[0]
    ds.model_output(model)

    print(ds.models)
    ds.instructions[:1]
    ds.get_instruction_index(ds.instructions[0])
    m = ds.model_output("gpt-4-1106-preview")
    for i in range(3):
        m[ds.get_instruction_index(ds.instructions[i])]

    print(ds.judges)

    print(ds.df_winrate_against_baseline(models=[model]))
    print(
        ds.df_winrate_against_baseline().mean().sort_values(ascending=False).to_string()
    )

    print(ds.num_annotations())
    print(ds.cost_per_annotation(models=[model]))
    print(ds.time_per_annotation(models=[model]))

    print(ds.elo_ratings().sort_values(ascending=False).to_string())
    print(ds.chatbot_arena_elo_ratings())

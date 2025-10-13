from ast import literal_eval
from pathlib import Path
import pandas as pd
import random
import numpy as np

from timeblock import Timeblock

from judgetuning.annotation_dataset.tables.table_annotation_dataset import (
    TableAnnotationDataset,
    table_root_default,
)
from judgetuning.utils import random_string

random.seed(0)
np.random.seed(0)


def import_lmsys_kaggle():

    name = "lmsys-kaggle"
    table_root = table_root_default
    lmsys_root = table_root / name
    lmsys_root.mkdir(exist_ok=True)

    lmsys_data_path = (
        Path(__file__).parent.parent.parent.parent / "lmsys-chatbot-arena/train.csv"
    )
    assert lmsys_data_path.exists(), f"{lmsys_data_path} does not exist."
    df = pd.read_csv(lmsys_data_path)

    # convert to instructions schema 'instruction', 'domain', 'dataset', 'instruction_index
    df_instruction = df.copy()
    df_instruction["dataset"] = name

    def multiturn_to_str(s: str):
        try:
            # instruction and outputs can be multiturn (a list), we map to a single string and use "\n" to split turns
            return "\n".join(literal_eval(s))
        except ValueError as e:
            # some expression are not well formatted
            return s

    df_instruction["instruction"] = df_instruction["prompt"].apply(multiturn_to_str)
    df_instruction.pop("prompt")

    # sort to be deterministic (set is not)
    instructions = list(sorted(set(df_instruction["instruction"])))

    def create_index(i: int, instruction: str) -> str:
        x = instruction[:20].replace(" ", "-").lower() + "...-" + f"{i}"
        return x.encode("utf-8", errors="replace").decode("utf-8")

    instruction_index_dict = {
        instruction: create_index(i, instruction)
        for i, instruction in enumerate(instructions)
    }
    df_instruction["instruction_index"] = df_instruction["instruction"].apply(
        lambda x: instruction_index_dict[x]
    )
    df_instruction["domain"] = name
    # df_instruction['instruction'] = df_instruction['instruction'].map(
    #     lambda x: x.encode('unicode-escape').decode('utf-8'))
    df_instruction = df_instruction.loc[
        :, ["dataset", "instruction", "instruction_index", "domain"]
    ]
    df_instruction.drop_duplicates(subset="instruction_index", inplace=True)
    df_instruction.to_csv(
        table_root / "instructions" / f"{name}.csv", index=False, errors="replace"
    )

    # conver to schema 'model', 'instruction_index', 'output'

    df_model_rows = []
    for i, row in df.iterrows():
        for model_letter in ["a", "b"]:
            df_model_rows.append(
                {
                    "model": row[f"model_{model_letter}"],
                    "instruction_index": instruction_index_dict[
                        multiturn_to_str(row["prompt"])
                    ],
                    "output": multiturn_to_str(row[f"response_{model_letter}"]),
                }
            )
    df_model = pd.DataFrame(df_model_rows)
    df_model["instruction_index"] = df["prompt"].apply(
        lambda x: instruction_index_dict[multiturn_to_str(x)]
    )
    df_model.to_csv(
        table_root / "model_outputs" / f"{name}.csv.zip", index=False, errors="replace"
    )
    df_judge_rows = []
    for i, row in df.iterrows():
        if row["winner_model_a"] == 1:
            preference = 0
        elif row["winner_model_b"] == 1:
            preference = 1
        else:
            assert row["winner_tie"] == 1
            preference = 0.5
        df_judge_rows.append(
            {
                "judge_index": "human",
                "model1": row[f"model_a"],
                "model2": row[f"model_b"],
                "output1": multiturn_to_str(row[f"response_a"]),
                "output2": multiturn_to_str(row[f"response_b"]),
                "instruction_index": instruction_index_dict[
                    multiturn_to_str(row["prompt"])
                ],
                "preference": preference,
                "prompt": "NA",
                "swap": False,
                "number_output_tokens": 0,
                "cost": 0,
                "time": 0,
            }
        )
    df_judge = pd.DataFrame(df_judge_rows)
    df_judge["output1"] = df_judge["output1"].map(
        lambda x: x.encode("utf-8", errors="replace").decode("utf-8")
    )
    df_judge["output2"] = df_judge["output2"].map(
        lambda x: x.encode("utf-8", errors="replace").decode("utf-8")
    )
    df_judge.to_parquet(
        table_root / "judge_annotations" / f"{name}.parquet", index=False
    )


class LMSysKaggleDataset(TableAnnotationDataset):
    def __init__(
        self,
        table_root: Path = table_root_default,
    ):
        super().__init__(
            name="lmsys-kaggle",
            table_root=table_root,
            instruction_filenames=["lmsys-kaggle.csv"],
            model_filenames=["lmsys-kaggle.csv.zip"],
            judge_filenames=["lmsys-kaggle.parquet"],
        )
        # if keep_only_chatbot_arena or rename_chatbot_arena:
        #     alpaca_eval_to_chatbot_arena = dict(
        #         load_model_mapping("alpaca-eval")
        #         .loc[:, ["alpaca-eval", "chatbot-arena"]]
        #         .values
        #     )
        #     if keep_only_chatbot_arena and not rename_chatbot_arena:
        #         self.subselect_models(alpaca_eval_to_chatbot_arena.keys())
        #     elif rename_chatbot_arena:
        #         self.rename_models(
        #             alpaca_eval_to_chatbot_arena, drop_missing=keep_only_chatbot_arena
        #         )
        print(self)

    @staticmethod
    def generate():
        import_lmsys_kaggle()


if __name__ == "__main__":
    LMSysKaggleDataset.generate()

    model = "gpt-4-0613"
    with Timeblock("Loading dataset") as t:
        ds = LMSysKaggleDataset()
    print([x for x in ds.instruction_index if "hello-bob" in x])
    ds.model_output(model)

    print(ds.models)
    ds.instructions[:1]
    print(ds.get_instruction_index(ds.instructions[0]))
    m = ds.model_output(model)
    for i in range(3):
        m[ds.get_instruction_index(ds.instructions[i])]

    print(ds.elo_ratings().sort_values().to_string())
    # print(ds.chatbot_arena_elo_ratings())

    ds.push_to_hf()

import json
from typing import List

import pandas as pd

from judgetuning.annotation_dataset import AnnotationDataset
from judgetuning.chatbot_arena_utils import compute_mle_elo
from judgetuning.join import chatbot_alpaca_arena_mapping
from judgetuning.judge_paths import path_annotation_model, path_annotation_judge


def load_judge_model_annotations(
    expid: str,
    instruction_dataset: str,
    judge_name: str,
    models: list[str],
    baseline: str | None = None,
):
    dfs = []
    path_judge = path_annotation_judge(expid, instruction_dataset, judge_name)
    assert path_judge.exists(), f"{path_judge} does not exists."
    if models is None:
        # no models given, compute the list of models from annotations found locally
        models = []
        for f in path_judge.glob("*"):
            if f.is_dir():
                model = f.name
                models.append(model)
    for model in models:
        model_path = path_annotation_model(
            expid=expid,
            instruction_dataset=instruction_dataset,
            judge_name=judge_name,
            model=model,
        )
        # assert model_path.exists(), (
        #     f"Files from {instruction_dataset}/{judge_name}/{model} are missing, (there should be at {model_path}).\n"
        #     f"Have you generated the judge annotations already?"
        # )
        annotations = []
        for model_annotations_path in model_path.rglob("annotations.json"):
            with open(model_annotations_path, "r") as f:
                for annotation in json.load(f):
                    assert len(annotation) > 0
                    annotations.append(annotation)
        dfs.append(pd.DataFrame(annotations))
    assert len(models) > 0
    df = pd.concat(dfs, ignore_index=True)
    if baseline is None:
        baselines = df.model1.unique()
        assert (
            len(baselines) == 1
        ), f"Expected exactly one baseline model but got {baselines}."
        baseline = baselines[0]
    else:
        df = df[df.generator_1 == baseline]
    return df, baseline


class JudgeTuningDataset(AnnotationDataset):
    def __init__(
        self,
        expid: str,
        judge_name: str,
        instruction_dataset: str,
        baseline: str | None = None,
        models: list[str] | None = None,
        rename_chatbot_arena: bool = False,
    ):
        super().__init__(name=f"judge-tuning-{instruction_dataset}")
        assert instruction_dataset.replace("_", "-") in ["alpaca-eval", "arena-hard"]
        self._df, self._baseline = load_judge_model_annotations(
            expid=expid,
            instruction_dataset=instruction_dataset,
            judge_name=judge_name,
            models=models,
            baseline=baseline,
        )
        # self._models = list(set(self._df.model2.unique().tolist() + self._df.model1.unique().tolist()))
        self._models = self._df.model2.unique().tolist()
        self._instructions_index = self._df.instruction_index.unique().tolist()

        self.rename_chatbot_arena = rename_chatbot_arena
        if rename_chatbot_arena:
            df_rename = chatbot_alpaca_arena_mapping()
            if "alpaca" in instruction_dataset:
                rename_dict = dict(df_rename.alpaca_eval)
            else:
                rename_dict = dict(df_rename.arena_hard)
            rename_dict = {v: k for k, v in rename_dict.items()}
            self._models = [rename_dict[x] for x in self._models]
            self._df = self._df.replace(
                {
                    "model1": rename_dict,
                    "model2": rename_dict,
                }
            )

        self._df_pivot = self._df.pivot_table(
            index="instruction_index",
            columns="model2",
            values=["preference", "cost", "time", "swap"],
            aggfunc="mean",
        )
        self._completions = self._df.pivot_table(
            # index=["instruction", "swap"],
            # TODO handle "swap"
            index="instruction_index",
            columns="model2",
            values="judge_completion",
            aggfunc="first",
        )
        self.judge_name = judge_name
        print(
            f"Loaded {len(self._df)} annotations of judge {judge_name} on expid={expid}: {len(self.models)} models (without "
            f"counting the baseline) on {len(self.instructions_index)} instructions. "
        )

    @property
    def models(self) -> list[str]:
        return self._models

    # TODO make it a list...
    def model_output(self, model: str) -> pd.Series:
        raise NotImplementedError()

    @property
    def instructions_index(self) -> list[str]:
        return self._instructions_index

    @property
    def judges(self) -> list[str]:
        return [self.judge_name]

    # def judge_annotation(
    #     self,
    #     models: list[str] | None = None,
    #     judge_name: str | None = None,
    #     instructions: list[str] | None = None,
    # ) -> pd.DataFrame:
    #     if instructions is None:
    #         instructions = self.instructions
    #     df = self._df_pivot.loc[
    #         instructions,
    #         (["preference", "cost", "time"], models),
    #     ]
    #     # df.columns = df.columns.droplevel(0)
    #     return df

    def judge_completions(
        self,
        models: list[str] | None = None,
        instructions: list[str] | None = None,
        judge_name: str | None = None,
    ) -> pd.DataFrame:
        assert judge_name is None or judge_name == self.judge_name
        if instructions is None:
            instructions = self.instructions_index
        instructions = [x for x in instructions if x in self._completions.index]
        models = [x for x in models if x in self._completions.columns]
        if models is None:
            models = self.models
        return self._completions.loc[instructions, models]

    def df_winrate_against_baseline(
        self,
        models: list[str] | None = None,
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        :param models:
        :param judge_name:
        :param instructions:
        :return: a table whose index/columns/values are the instructions/models and preference values in [0, 1]
        against the baseline (higher is better for the model)
        """
        if models is None:
            models = self.models
        if instructions is None:
            instructions = self.instructions_index
        df = self._sub_df(models, instructions)
        return df.pivot_table(
            index="instruction_index", columns="model2", values="preference"
        ).loc[:, models]

    def elo_ratings(
        self,
        models: list[str] | None = None,
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> pd.Series:
        def winner_from_pref(preference: float) -> str:
            if preference < 0.5:
                return "model_a"
            elif preference > 0.5:
                return "model_b"
            else:
                return "tie"

        if models is None:
            models = self.models
        if instructions is None:
            instructions = self.instructions_index
        sub_df = self._sub_df(models, instructions)
        df_battles = sub_df.loc[
            :, ["instruction_index", "model1", "model2", "preference"]
        ]
        df_battles = df_battles.rename(
            columns={"model1": "model_a", "model2": "model_b"}
        )
        df_battles["winner"] = df_battles.apply(
            lambda x: winner_from_pref(x["preference"]), axis=1
        )
        return compute_mle_elo(df_battles)

    def cost_per_annotation(
        self,
        models: list[str] | None = None,
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> float:
        return float(self._sub_df(models, instructions).cost.mean())

    def time_per_annotation(
        self,
        models: list[str] | None = None,
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> float:
        return float(self._sub_df(models, instructions).time.mean())

    def num_annotations(
        self,
        models: list[str] | None = None,
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> int:
        return len(self._sub_df(models, instructions))

    def _sub_df(
        self,
        models: list[str] | None = None,
        instruction_index: list[str] | None = None,
    ):
        if instruction_index is None:
            instruction_index = self.instruction_index
        if models is None:
            models = self.models
        mask = (self._df.model2.isin(models)) & (
            self._df.instruction_index.isin(instruction_index)
        )
        return self._df.loc[mask, :]

    @property
    def instruction_index(self) -> list[str]:
        return self._instructions_index


if __name__ == "__main__":
    judge_annotation = JudgeTuningDataset(
        expid="yop",
        judge_name="yop_gpt-4o-2024-05-13_judge-arena-hard_1_arena-hard_4096_20_5_True_0_0_0_0_pair",
        instruction_dataset="arena_hard",
        rename_chatbot_arena=True,
    )

    print("num annotations", str(judge_annotation.num_annotations()))
    print(judge_annotation.cost_per_annotation())
    print(judge_annotation.elo_ratings())
    print(judge_annotation.chatbot_arena_elo_ratings())

    df = judge_annotation.df_winrate_against_baseline()

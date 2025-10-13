from typing import List

import pandas as pd

from judgetuning.chatbot_arena_utils import load_chatbot_arena_elo_ratings


class AnnotationDataset:
    def __init__(self, name: str):
        """
        Abstract class representing an annotation dataset that contains instructions, model outputs and judge
        annotations. Annotations are available for AlpacaEval, ArenaHard, LLMBarb, LMSys and PandaLM, see `tables/`.
        :param name:
        """
        self.name = name

    def __str__(self) -> str:
        n_models = len(self.models)
        n_instructions = len(self.instructions)
        elo = self.elo_ratings().sort_values(ascending=False)
        top = elo.index[0]
        worse = elo.index[-1]
        return f"{self.name} dataset with {n_models} models and {n_instructions} instructions. Top model {top}, worse model {worse}."

    @property
    def models(self) -> List[str]:
        raise NotImplementedError()

    def model_output(self, model: str) -> pd.Series:
        raise NotImplementedError()

    # TODO important: the API has been made dissonant by naming instruction/instruction_index intercheangeably
    @property
    def instructions(self) -> list[str]:
        raise NotImplementedError()

    @property
    def instruction_index(self) -> list[str]:
        raise NotImplementedError()

    def get_instruction_index(self, instruction: str) -> str:
        return self.instructions_index.index(instruction)

    @property
    def judges(self) -> list[str]:
        raise NotImplementedError()

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
        raise NotImplementedError()

    def chatbot_arena_elo_ratings(
        self,
        models: list[str] | None = None,
    ) -> pd.Series:
        if models is None:
            models = self.models
        # return load_chatbot_arena_elo_ratings().set_index("model_name").loc[models]
        cb_elo_ratings = load_chatbot_arena_elo_ratings()
        for m in models:
            if m not in cb_elo_ratings:
                print(f"WARNING: model {m} not in chatbot_arena, skipping it.")
        return cb_elo_ratings.loc[[m for m in models if m in cb_elo_ratings]]

    def elo_ratings(
        self,
        models: list[str] | None = None,
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> pd.Series:
        raise NotImplementedError()

    def cost_per_annotation(
        self,
        models: list[str],
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> float:
        raise NotImplementedError()

    def time_per_annotation(
        self,
        models: list[str],
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> float:
        raise NotImplementedError()

    def num_annotations(
        self,
        models: list[str] | None = None,
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> int:
        raise NotImplementedError()

    def judge_annotation(
        self,
        judge_name: str,
        models: list[str] | None = None,
        instructions: list[str] | None = None,
    ):
        pass

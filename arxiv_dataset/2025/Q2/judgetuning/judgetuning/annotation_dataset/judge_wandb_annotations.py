import json
import numpy as np
import pandas as pd
import wandb
from scipy.stats import spearmanr

from judgetuning.annotation_dataset.judge_tuning_dataset import JudgeTuningDataset
from judgetuning.join import chatbot_alpaca_arena_mapping, load_model_mapping
from judgetuning.judge_paths import judge_tuning_datadir
from pathlib import Path


def retrieve_table(path: str, name: str, run_id: str | None):
    def retrieve_run(path: str, name: str, run_id: str | None) -> wandb.apis.public.Run:
        api = wandb.Api()
        runs = api.runs(
            path=path,
            #    filters={"tags": {"$in": [tag]}}
        )
        for run in runs:
            if run.name == name and (run.id == run_id or run_id is None):
                # print("got it!cd -")
                return run
        raise ValueError(f"Job {name} not found.")

    _run = retrieve_run(path=path, name=name, run_id=run_id)
    table_name = "judgements"
    if len(_run.logged_artifacts()) == 0:
        raise ValueError(f"No table found for job {name}.")
    table = _run.logged_artifacts()[0]
    assert table_name in table.name, f"Wrong table name for judge {table.name}"

    table_dir = table.download(judge_tuning_datadir / "wandb" / (name + "-" + run_id))
    table_path = f"{table_dir}/judgements.csv.zip"
    return (
        pd.read_csv(table_path),
        _run.config,
        _run.summary,
    )


def make_wandb_annotations(
    job_name: str,
    run_id: str,
    ignore_cache: bool = False,
    path: str = "geoalgo-university-of-freiburg/judge-evaluations",
):
    judgement_path = (
        judge_tuning_datadir
        / "wandb"
        / (job_name + "-" + run_id)
        / "judgements.csv.zip"
    )
    run_config_path = (
        judge_tuning_datadir / "wandb" / (job_name + "-" + run_id) / "run_config.json"
    )
    run_summary_path = (
        judge_tuning_datadir / "wandb" / (job_name + "-" + run_id) / "run_summary.json"
    )
    run_summary_path.parent.mkdir(exist_ok=True, parents=True)
    if not ignore_cache and judgement_path.exists():
        df_annotations = pd.read_csv(judgement_path)
        with open(run_config_path, "r") as f:
            run_config = json.load(f)
        with open(run_summary_path, "r") as f:
            run_summary = json.load(f)
    else:
        df_annotations, run_config, run_summary = retrieve_table(
            path=path,
            name=job_name,
            run_id=run_id,
        )
        # save run_config and run_summary to json
        with open(run_config_path, "w") as f:
            json.dump(run_config, f)
        with open(run_summary_path, "w") as f:
            # avoids error TypeError: Object of type HTTPSummary is not JSON serializable
            json.dump(
                {k: v for k, v in run_summary.items() if isinstance(v, (int, float))},
                f,
            )
    return JudgeWandbAnnotations(
        name=job_name,
        run_id=run_id,
        df_annotations=df_annotations,
        run_config=run_config,
        run_summary=run_summary,
    )


class JudgeWandbAnnotations(JudgeTuningDataset):
    def __init__(
        self,
        name: str,
        run_id: str,
        df_annotations: pd.DataFrame,
        run_config: dict,
        run_summary: dict,
    ):
        self.name = name
        self.run_id = run_id
        self._df = df_annotations
        self._run_config = run_config
        # self._run_summary = run_summary
        # serializing run_summary causes weird recursion error, we just track the spearman correlation
        self._spearman_correlation = run_summary.get("spearman_correlation", np.nan)
        self._instructions_index = self._df.instruction_index.unique().tolist()
        self._models = self._df.model2.unique().tolist()
        self._judge_name = run_config["jobname"]

    def spearman_correlation(self):
        return self._spearman_correlation

    def __str__(self) -> str:
        n_models = len(self.models)
        n_instructions = len(self.instructions)
        elo = self.elo_ratings().sort_values(ascending=False)
        top = elo.index[0]
        worse = elo.index[-1]
        return f"{self.name} dataset with {n_models} models and {n_instructions} instructions. Top model {top}, worse model {worse}."

    def model_output(self, model: str) -> pd.Series:
        raise NotImplementedError()

    @property
    def instructions(self) -> list[str]:
        return self._instructions

    @property
    def models(self) -> list[str]:
        if not hasattr(self, "_models"):
            self._models = self._df.model2.unique().tolist()
        return self._models

    @property
    def instruction_index(self) -> list[str]:
        return self._instructions_index

    @property
    def judges(self) -> list[str]:
        return [self.judge_name]

    def judge_annotation(
        self,
        judge_name: str,
        models: list[str] | None = None,
        instructions: list[str] | None = None,
    ):
        pass

    # def __getstate__(self):
    #     return {
    #         "name": self.name,
    #         "path": self.path,
    #         "df": self._df,
    #         "run_config": self._run_config,
    #         "run_summary": self._run_summary,
    #     }
    #
    # def __setstate__(self, state):
    #     self.name = state["name"]
    #     self.path = state["path"]
    #     self._df = state["df"]
    #     self._run_config = state["run_config"]
    #     self._run_summary = state["run_summary"]


if __name__ == "__main__":
    from timeblock import Timeblock

    with Timeblock("Loading judge") as t:
        judge_annotation = make_wandb_annotations(
            job_name="judge-tuning-v8-loop/judge-option-2024-10-12-20-45-08",
            ignore_cache=True,
        )

    with Timeblock("Loading judge with cache") as t:
        judge_annotation = make_wandb_annotations(
            job_name="judge-tuning-v8-loop/judge-option-2024-10-12-20-45-08",
        )

    print("num annotations", str(judge_annotation.num_annotations()))
    print(judge_annotation.models)
    print(judge_annotation.cost_per_annotation())
    print(judge_annotation.elo_ratings())
    elo_chatbot_arena = judge_annotation.chatbot_arena_elo_ratings()
    winrate = judge_annotation.df_winrate_against_baseline().mean()
    print(
        "spearmanr",
        spearmanr(elo_chatbot_arena, winrate.loc[elo_chatbot_arena.index]).statistic,
    )

    import pickle

    # save judge to pickle
    with open("judge_annotation.pkl", "wb") as f:
        pickle.dump(judge_annotation, f)

    # load judge from pickle
    with open("judge_annotation.pkl", "rb") as f:
        judge_annotation = pickle.load(f)

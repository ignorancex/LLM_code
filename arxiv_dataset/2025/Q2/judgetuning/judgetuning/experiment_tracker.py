import pandas as pd
import wandb
import tempfile
from pathlib import Path
import pickle
from judgetuning.judge import Judge, JudgeAnnotation
from judgetuning.judge.eval_judge import JudgeEvaluationMetrics
from judgetuning.judge.io.judge_arena_hard import JudgeArenaHard
from time import time

default_project_name = "judge-evaluations"


class ExperimentTracker:
    def __init__(
        self,
        judge: Judge,
        instructions: list[str],
        models: list[str],
        project_name: str | None = None,
        instruction_dataset: str | None = None,
        tags: list[str] | None = None,
        jobname: str | None = None,
    ):
        """
        Initialize an ExperimentTracker.

        Args:
        judge (Judge): The judge to be evaluated.
        judge_name (str): The name of the judge.
        instructions (list[str]): The instructions to be evaluated.
        models (list[str]): The models to be evaluated.
        project_name (str | None, optional): The name of the W&B project to log the experiment to. Defaults to project_name.
        instruction_dataset (str | None, optional): The name of the instruction dataset. Defaults to None.
        tags (list[str] | None, optional): The tags for the W&B run. Defaults to None.

        The ExperimentTracker will log the judge JSON, instruction dataset, number of instructions,
        number of models, and completion client JSON to the W&B run config. It will also log the
        experiment to the W&B project specified by `project_name`.

        We create a class to be able to abstract to other tracking library, also possibly implementing our
        local storage with this framework.
        """
        if project_name is None:
            project_name = default_project_name
        self.judge = judge
        self.instruction_dataset = instruction_dataset
        self.instructions = instructions
        self.models = models
        n_models = len(models)
        n_instructions = len(instructions)
        wandb.init(
            entity="geoalgo-university-of-freiburg",
            project=project_name,
            name=jobname,
            # wandb only supports 64 character tags but it only tell you afterwards if you run offline
            tags=[x[:64] for x in tags] if tags else None,
            config={
                "judge_json": judge.to_json(),
                "instruction_dataset": instruction_dataset,
                "n_instructions": n_instructions,
                "n_models": n_models,
                "jobname": jobname,
                "completion_client": (
                    judge.completion_client.to_json()
                    if hasattr(judge, "completion_client")
                    else None
                ),
            },
            group=f"{instruction_dataset}-{n_models}-{n_instructions}",
        )

    def __del__(self):
        try:
            wandb.finish()
        except Exception as e:
            print(str(e))

    def _save_df(self, df: pd.DataFrame, name: str):
        with tempfile.TemporaryDirectory() as tmpdirname:
            df.to_csv(
                Path(tmpdirname) / f"{name}.csv.zip", escapechar="\\", index=False
            )
            wandb.log_artifact(Path(tmpdirname) / f"{name}.csv.zip", name)

    def track_judge_annotations(self, judge_annotations: list[JudgeAnnotation]):
        self._save_df(pd.DataFrame(judge_annotations), "judgements")
        # wandb.log({"judgements": df})

    def track_judge_annotations_and_human_preference(
        self,
        judge_annotations: list[list[JudgeAnnotation]],
        human_preferences: list[float],
    ):
        assert len(judge_annotations) == len(human_preferences)
        rows = []
        for human_preference, judge_annotation in zip(
            human_preferences, judge_annotations
        ):
            for x in judge_annotation:
                rows.append(x.__dict__)
                rows[-1]["human_preference"] = human_preference
        df = pd.DataFrame(rows)
        print(df.columns)
        # wandb.log({"judgements": df})
        try:
            self._save_df(df, "judgements")
        except Exception as e:
            print("Got exception while saving to wandb" + str(e))
            print("Saving annotations instead locally to judgements.csv.zip")
            df.to_csv("judgements.csv.zip", escapechar="\\", index=False)

    def track_results(self, results: JudgeEvaluationMetrics):
        wandb.log(results.__dict__)

    def track_human_agreement(self, human_agreement: float):
        wandb.log({"human_agreement": human_agreement})


if __name__ == "__main__":
    from judgetuning.llm_client import OllamaCompletionClient

    completion_client = OllamaCompletionClient(model="yop")
    instructions = ["do this", "do that"]
    models = ["llama3", "bird2", "cat5"]
    tracker = ExperimentTracker(
        project_name="judge-evaluation-dummy",
        judge=JudgeArenaHard(completion_client=completion_client),
        instructions=instructions,
        models=models,
        tags=["test"],
        instruction_dataset="alpaca-eval",
    )

    judge_annotations = []
    for model in models:
        for instruction in instructions:
            judge_annotation = JudgeAnnotation(
                preference=0.2,
                instruction_index=instruction,
                output1=f"I am {model}. You asked me {instruction}",
                output2=f"You asked me {instruction} dezodhez",
                model1=model,
                model2="baseline",
                prompt=instruction,
                judge_completion="Both output are bad.",
            )
            judge_annotations.append(judge_annotation)

    tracker.track_judge_annotations(judge_annotations=judge_annotations)

    tracker.track_results(
        results=JudgeEvaluationMetrics(
            spearman_correlation=0.6812030075187969,
            time_per_annotation=1.6311701810359955,
            cost_per_annotation=123.0,
            time_total=50.0,
            cost_total=12.0,
            n_annotations=len(instructions),
            n_missing=0,
            avg_length=12,
            n_empty_completions=0,
        )
    )

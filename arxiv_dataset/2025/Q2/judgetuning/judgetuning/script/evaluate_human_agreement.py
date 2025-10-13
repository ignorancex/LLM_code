import os

import numpy as np

from judgetuning.annotation_dataset import LMSysKaggleDataset
from judgetuning.annotation_dataset.tables.tables_llmbar import LLMBarDataset
from judgetuning.annotation_dataset.tables.tables_pandalm import PandaLMDataset
from judgetuning.experiment_tracker import ExperimentTracker
from judgetuning.judge.generate_judge_annotations import RequestInfo
from judgetuning.script.utils import (
    make_judge,
    parse_args,
)
from judgetuning.script.select_lmsys_instructions import val_test_split


def error_judgement(y_true, y_pred):
    if y_true < 0.5:
        return y_pred < 0.5
    elif y_true == 0.5:
        return y_pred == 0.5
    else:
        return y_pred > 0.5


if __name__ == "__main__":
    jobname = os.getenv("SP_JOBNAME", "")
    wandb_project_name = "human-agreement"
    shuffle_seed = 0
    args = parse_args()
    num_annotations = args.n_instructions

    # 1) load annotation dataset

    print(f"Using dataset {args.dataset} on {args.split} split")
    if args.dataset == "pandalm":
        assert args.split == "test", "only the test set of pandalm is available"
        ds = PandaLMDataset()
        instructions_index = ds.instruction_index
    elif args.dataset == "llmbar":
        assert args.split == "test", "only the test set of llmbar is available"
        ds = LLMBarDataset()
        instructions_index = ds.instruction_index
    else:
        ds = LMSysKaggleDataset()
        val_instructions, test_instructions = val_test_split()
        instructions = val_instructions if args.split == "val" else test_instructions
        instructions_index = [ds.get_instruction_index(x) for x in instructions]

    print(
        f"Going to use {len(instructions_index)} instructions (before filtering `n_instructions`)"
    )
    df = ds._df_judge
    df = df.loc[df.loc[:, "instruction_index"].isin(instructions_index), :]

    # some instruction may have several associated battles, we pick only one in such cases
    if args.dataset == "pandalm":
        print("Only using mode annotation")
        df = df[df["judge_index"] == "annotator-mode"]
    else:
        df = df.drop_duplicates(subset="instruction_index", inplace=False)

    judge = make_judge(args)

    print(f"Going to evaluate {judge.to_json()}")

    instructions_tracked = df.loc[:, "instruction_index"].to_list()
    if num_annotations is not None:
        instructions_tracked = instructions_tracked[:num_annotations]
    experiment_tracker = ExperimentTracker(
        project_name=wandb_project_name,
        judge=judge,
        instructions=instructions_tracked,
        instruction_dataset=ds.name,
        models=[f"{x}-{y}" for x, y in zip(df.loc[:, "model1"], df.loc[:, "model2"])],
        jobname=jobname,
        tags=[args.expid],
    )

    # 2) generate annotation requests
    requests = []
    human_preferences = []
    rows = list(df.to_dict(orient="records"))
    if num_annotations is not None:
        rows = rows[:num_annotations]
    for row in rows:
        requests.append(
            RequestInfo(
                instruction_index=row["instruction_index"],
                instruction=ds.get_instruction(row["instruction_index"]),
                model1=row["model1"],
                model2=row["model2"],
                output1=row["output1"],
                output2=row["output2"],
            )
        )
        human_preferences.append(row["preference"])

    print(f"Going to annotate {len(requests)} battles.")

    # 3) generate the judge annotation for the given judge
    judge_annotations = judge.annotate(requests)

    # 4) compute human-agreement
    judge_preferences = []
    cost = 0
    for annotation_games in judge_annotations:
        avg_preference = np.mean(
            [annotation.preference for annotation in annotation_games]
        )
        judge_preferences.append(avg_preference)
        cost += np.sum([annotation.cost for annotation in annotation_games])

    print(f"Cost: {cost}$")

    human_agreement = np.mean(
        [error_judgement(x, y) for x, y in zip(human_preferences, judge_preferences)]
    )
    print(f"Human agreement: {human_agreement}")

    experiment_tracker.track_judge_annotations_and_human_preference(
        human_preferences=human_preferences, judge_annotations=judge_annotations
    )
    experiment_tracker.track_human_agreement(human_agreement)

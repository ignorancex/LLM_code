from pathlib import Path

judge_tuning_datadir = Path("~/judge-tuning-data/").expanduser()
judge_tuning_datadir.mkdir(exist_ok=True, parents=True)

experiments_dir = judge_tuning_datadir / "experiments"
experiments_dir.mkdir(exist_ok=True, parents=True)

annotation_dir = "annotation"


def path_exp_dir(expid: str) -> Path:
    return experiments_dir / expid


def path_judge(expid: str, judge_name: str):
    return path_exp_dir(expid) / judge_name


def path_config_judge(expid: str, judge_name: str) -> Path:
    return path_judge(expid, judge_name) / "config.json"


def path_result_judge(expid: str, judge_name: str) -> Path:
    return path_judge(expid, judge_name) / "results.json"


def path_annotation_judge(
    expid: str,
    instruction_dataset: str,
    judge_name: str,
) -> Path:
    return path_judge(expid, judge_name) / annotation_dir / instruction_dataset


def path_annotation_model(
    expid: str,
    instruction_dataset: str,
    judge_name: str,
    model: str,
) -> Path:
    return path_annotation_judge(expid, instruction_dataset, judge_name) / model


def path_annotation_instruction(
    expid: str,
    instruction_dataset: str,
    judge_name: str,
    model: str,
    instruction_index: int | str,
) -> Path:
    return (
        path_annotation_model(expid, instruction_dataset, judge_name, model)
        / str(instruction_index).replace("/", "-")
        / "annotations.json"
    )

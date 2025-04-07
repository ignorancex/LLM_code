import json
from collections import defaultdict, Counter
from copy import copy
from typing import Dict, Tuple, List

from src.common import flatten
from omegaconf import OmegaConf
from promptsource.templates import Template
import wandb
from scipy.special import softmax
import pandas as pd
import numpy as np
from src.common import sanitize_name


def get_prompt_info_for_wandb(
        task_name: str,
        prompt_group: str,
        prompt: Template,
        prompt_metadata: Dict,
) -> Tuple[List, Dict]:
    prompt_cfg = {
        "choices_in_prompt"   : prompt.metadata.choices_in_prompt or False,
        "original_task"       : prompt_metadata.get("original_task", prompt.metadata.original_task),
        "has_choices"         : prompt.answer_choices is not None,
        "prompt_name"         : prompt_metadata['name'],
        "original_prompt_name": prompt.name,
        "prompt_id"           : prompt.id,
        "prompt_category"     : prompt_metadata.get("category", None),
        "prompt_group"        : prompt_group,
        "is_mcq"              : prompt_metadata.get("is_mcq", None),
        "task_mode"           : prompt_metadata.get("task_mode", None),
        "training_task"       : prompt_metadata.get('training_task', False)
    }

    choices = prompt.get_fixed_answer_choices_list()
    choice_count = len(choices if choices else 0)

    if choices is not None:
        choice_str = " | ".join(choices)
    else:
        choice_str = "N/A"

    prompt_cfg["choices"] = choice_str
    prompt_cfg["choice_count"] = choice_count
    prompt_cfg['prompt_task'] = prompt_metadata['prompt_task']

    tags = [
        f"PromptCat:{prompt_cfg['prompt_category']}",
        prompt_cfg['prompt_group']
    ]
    return tags, prompt_cfg


def create_run_cfg(
        cfg,
        prompt_group,
        prompt,
        prompt_metadata
) -> Tuple[List, Dict]:
    cfg_dict = OmegaConf.to_object(cfg)
    cfg_dict['task']['general_prompts'] = None
    run_cfg = {
        "base_model"          : cfg['evaluation'].get('base_model', False),
        "length_normalization": cfg['evaluation'].get('length_normalization'),
        "force_generation"    : cfg['evaluation'].get('force_generation'),
        **flatten(cfg_dict, sep='.')
    }

    tags, prompt_cfg = get_prompt_info_for_wandb(
        task_name=cfg['task']['name'],
        prompt_group=prompt_group,
        prompt=prompt,
        prompt_metadata=prompt_metadata
    )

    run_cfg.update(prompt_cfg)
    return tags, run_cfg


def get_metrics_for_wandb(metrics_path, predictions_path, choices):
    records = []
    metrics = json.loads(metrics_path.read_text('utf-8'))
    for line in predictions_path.read_text('utf-8').splitlines(False):
        if not line:
            continue
        line_record = json.loads(line)
        line_record['correct'] = line_record['prediction'] == line_record['target']

        for choice, (choice_id, logit) in zip(choices,
                                              line_record.pop('choice_logits').items()):
            line_record[f"choice_{choice_id}"] = choice
            line_record[f"choice_{choice_id}_logit"] = logit
        records.append(line_record)
    return metrics, pd.DataFrame.from_records(records).sort_values(
        by=['id']
    )


def save_run_to_wandb(
        run_name,
        run_cfg,
        metrics,
        pred_df,
        tags,
        categories,
        group_name,
        name,
        metrics_path,
        predictions_path,
        is_debug
):
    assert metrics_path.exists()
    assert predictions_path.exists()

    tags.extend(categories)

    pred_table = wandb.Table(dataframe=pred_df)
    wandb_run = wandb.init(
        project=f"{'debug-' if is_debug else ''}zero-shot-eval",
        job_type="evaluation" if not is_debug else "debugging",
        entity="gabeorlanski",
        group=group_name,
        name=name,
        tags=tags,
        config=run_cfg
    )
    wandb_run.log(metrics)

    wandb_run.log({"predictions": pred_table})
    artifact = wandb.Artifact(f"{sanitize_name(group_name)}.{sanitize_name(run_name)}",
                              'predictions')
    artifact.add_dir(metrics_path.parent)
    wandb_run.log_artifact(artifact)

    wandb_run.finish()

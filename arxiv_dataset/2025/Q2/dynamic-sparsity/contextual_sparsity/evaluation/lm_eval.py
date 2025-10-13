# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from transformers import PreTrainedModel, PreTrainedTokenizer

from contextual_sparsity.hw_simulator.simulator import HardwareSimulator

log = logging.getLogger(__name__)


def make_dataframe(out: Dict[str, Any]) -> pd.DataFrame:
    """
    Make a dataframe form the lm_eval output
    """
    pd_results = []
    for metric_name, stat_values in out["results"].items():
        version = out["versions"].get(metric_name, "N/A")
        n = str(out["n-shot"].get(metric_name, "N/A"))

        if "alias" in stat_values:
            metric_name = stat_values.pop("alias")

        for stat, value in stat_values.items():
            stat, _, f = stat.partition(",")

            if stat == "acc":
                stat = "mean"
            if stat == "acc_stderr":
                stat = "std"
            if stat == "acc_norm":
                stat = "norm"
            if stat == "acc_norm_stderr":
                stat = "norm_std"

            pd_results.append(
                {
                    "quantity": metric_name,
                    "stat": stat,
                    "value": value,
                    "version": version,
                    "filter": f,
                    "n_shots": n,
                }
            )
    return pd.DataFrame(pd_results)


def run_lm_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tasks: List[Any],
    store_full_output: bool = False,
    hw_simulator: Optional[HardwareSimulator] = None,
    **kwargs,
):
    """
    Run the lm_eval evaluation on a specified model
    """
    if not isinstance(tasks, list):
        tasks = list(tasks)

    if "limit" in kwargs:
        log.warning(
            f"The evaluation is running with limit={kwargs['limit']}. Use this only while debugging!"
        )

    # Wrap it into a HFLM object for evaluation
    lm_model = HFLM(model, tokenizer=tokenizer)

    # Run the simple_eval method from lm_eval
    out = evaluator.simple_evaluate(lm_model, tasks=tasks, **kwargs)
    log.info("\n" + make_table(out))

    # Store the full output if specified
    if store_full_output:
        out_path = os.path.abspath(os.path.join("", "lm_eval_full.yaml"))
        # Delete the configuration since it is not serializable
        del out["config"]
        with open(out_path, "w") as f:
            yaml.dump(out, f)
        log.info(f"Results stored in {out_path}")

    # Convert the results to a dataframe and store it
    pd_results = make_dataframe(out)
    summary_path = os.path.abspath(os.path.join("", "lm_eval_results.csv"))
    log.info(f"Summary results stored in {summary_path}")
    pd_results.to_csv(summary_path, index=False)

    # Get and store results from HW simulator
    if hw_simulator is not None:
        results_hwsim = hw_simulator.get_stats_df()
        log.info(results_hwsim)
        results_hwsim.to_csv("results_hwsim.csv", index=False)
        log.info(
            f"Results for HW simulator saved in {os.path.abspath(os.path.join('', 'results_hwsim.csv'))}"
        )

    average_accuracy = pd_results[pd_results["stat"] == "mean"]["value"].mean()
    log.info(f"Average Accuracy: {average_accuracy}")

    return {"average_accuracy": average_accuracy}

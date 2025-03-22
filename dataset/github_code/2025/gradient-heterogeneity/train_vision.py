import json
import logging
import os
import sys

import torch

from utils_shared.run_utils_shared import get_log_dir, parse_cli_args
from utils_vision.run_utils_vision import (
    calculate_accuracy,
    get_optimizer,
    setup,
    train_model,
)

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    cli_args = parse_cli_args()
    model, criteria, data_loaders, dataset_sizes, eps = setup(cli_args)

    optimizer = get_optimizer(cli_args, model)

    sched = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=eps * 0.1)

    log_dir = get_log_dir(cli_args)
    if (
        os.path.exists(os.path.join(log_dir, "all_results.json"))
        and not cli_args.overwrite
    ):
        logger.info("Already trained")
        return

    result_metrics = train_model(
        model,
        criteria,
        optimizer,
        sched,
        data_loaders,
        dataset_sizes,
        cli_args,
        num_epochs=eps,
        log_dir=log_dir,
    )

    model_best, model_last = (
        result_metrics["model_best"].cpu(),
        result_metrics["model_last"].cpu(),
    )
    checkpoint = {
        "arch": cli_args.model_name,
        "state_dict_best": model_best.state_dict(),
        "state_dict_last": model_last.state_dict(),
    }
    torch.save(
        checkpoint,
        f"{log_dir}/checkpoint.pth",
    )

    accuracy_metrics = calculate_accuracy(model_best, "test", data_loaders, cli_args)

    predict_result = {
        "eval_accuracy": result_metrics["eval_accuracy"],
        "train_loss": result_metrics["train_loss"],
    }
    for key in accuracy_metrics:
        predict_result[f"test_{key}"] = accuracy_metrics[key]
    with open(os.path.join(log_dir, "all_results.json"), "w") as f:
        json.dump(predict_result, f)
    logger.info("Done")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    _default_log_level = logging.INFO
    logger.setLevel(_default_log_level)
    main()

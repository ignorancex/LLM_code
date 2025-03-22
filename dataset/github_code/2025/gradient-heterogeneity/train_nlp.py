import logging
import os
import sys
from functools import partialmethod

import tqdm
import yaml

from utils_nlp.run_utils_nlp import (
    check_output_dir,
    evaluate,
    predict,
    prepare_args,
    setup_library,
    train,
)
from utils_nlp.tasks.get_trainer import get_trainer
from utils_shared.run_utils_shared import get_hyperparameter_config_path, parse_cli_args

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

logger = logging.getLogger(__name__)


def main():
    cli_args = parse_cli_args()
    with open(cli_args.arguments, "r") as f:
        config = yaml.safe_load(f)
    model_args, data_args, training_args = prepare_args(config, cli_args)
    hyperparameter_config_path = get_hyperparameter_config_path(cli_args)
    if not os.path.exists(hyperparameter_config_path):
        logger.info(
            f"No hyperparameter config found: {cli_args}, {hyperparameter_config_path}"
        )
        return
    with open(hyperparameter_config_path, "r") as f:
        hyperparameter_config = yaml.safe_load(f)
    training_args.learning_rate = hyperparameter_config[data_args.dataset_name][
        training_args.optimizer_name
    ]["lr"]
    training_args = setup_library(training_args)
    to_continue = check_output_dir(training_args)
    if not to_continue:
        return
    trainer, predict_dataset = get_trainer((model_args, data_args, training_args))
    logger.info(
        f"Start {data_args.dataset_name} dataset with training_mode {training_args.training_mode}, and seed {training_args.seed}"
    )
    if training_args.do_train:
        train(trainer)
    if training_args.do_eval:
        evaluate(trainer)
    if training_args.do_predict:
        predict(trainer, predict_dataset)
    logger.info("All done!")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    _default_log_level = logging.INFO
    logger.setLevel(_default_log_level)

    main()

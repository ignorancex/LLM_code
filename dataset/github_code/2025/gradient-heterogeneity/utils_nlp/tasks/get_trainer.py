import logging
import math
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    set_seed,
)

from model.utils import TaskType, get_model
from optimizer.signum import Signum
from utils_nlp.tasks.glue.dataset import GlueDataset
from utils_nlp.tasks.superglue.dataset import SuperGlueDataset

logger = logging.getLogger(__name__)


def get_optimizer(model, training_args):
    if training_args.optimizer_name == "sgd":
        logger.info("Use SGD.")
        optimizer = torch.optim.SGD(
            model.parameters(), lr=training_args.learning_rate, momentum=0.0
        )
    elif training_args.optimizer_name == "sgd_momentum":
        logger.info("Use SGD with momentum.")
        optimizer = torch.optim.SGD(
            model.parameters(), lr=training_args.learning_rate, momentum=0.9
        )
    elif training_args.optimizer_name == "signsgd":
        logger.info(f"Use SignSGD")
        optimizer = Signum(
            model.parameters(),
            lr=training_args.learning_rate,
            momentum=0.0,
        )
    elif training_args.optimizer_name == "signsgd_momentum":
        logger.info(f"Use SignSGD with momentum")
        optimizer = Signum(
            model.parameters(),
            lr=training_args.learning_rate,
            momentum=0.9,
        )
    elif training_args.optimizer_name == "adam":
        logger.info("Use Adam.")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_args.learning_rate,
            eps=training_args.adam_epsilon,
        )
    elif training_args.optimizer_name == "rmsprop":
        logger.info("Use RMSprop.")
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=training_args.learning_rate
        )
    else:
        raise ValueError(f"Optimizer {training_args.optimizer_name} not supported.")
    return optimizer


def get_scheduler(optimizer, training_args, dataset):
    num_training_steps = (
        math.ceil(
            len(dataset.train_dataset)
            / (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
            )
        )
        * training_args.num_train_epochs
    )
    if training_args.lr_scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif training_args.lr_scheduler_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=training_args.warmup_steps
        )
    elif training_args.lr_scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif training_args.lr_scheduler_type == "linear_with_warmup":
        num_warmup_steps = num_training_steps * 0.1
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        raise ValueError(f"Scheduler {training_args.lr_scheduler_type} not supported.")


def get_trainer(args):
    model_args, data_args, training_args = args
    logger.info(f"set model random seed {model_args.model_seed}")
    set_seed(model_args.model_seed)
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    if data_args.task_name.lower() == "superglue":
        dataset = SuperGlueDataset(tokenizer, data_args, training_args)
    elif data_args.task_name.lower() == "glue":
        dataset = GlueDataset(tokenizer, data_args, training_args)
    else:
        raise ValueError(f"Task {data_args.task_name} not supported.")

    assert not (
        hasattr(dataset, "multiple_choice") and dataset.multiple_choice
    ), "Multiple choice not supported"
    assert not (
        hasattr(dataset, "is_regression") and dataset.is_regression
    ), "Regression not supported"
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=dataset.num_labels,
        label2id=dataset.label2id,
        id2label=dataset.id2label,
        finetuning_task=data_args.dataset_name,
        revision=model_args.model_revision,
    )
    assert not hasattr(config, "adapter_config"), "Adapter not supported"
    config.lora = False
    task_type = TaskType.SEQUENCE_CLASSIFICATION
    model = get_model(
        model_args,
        task_type,
        config,
        fix_bert=False,
    )
    set_seed(training_args.seed)
    logger.info(f"set data random seed {model_args.model_seed}")

    callbacks = []
    if (not training_args.no_early_stopping) and training_args.load_best_model_at_end:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=model_args.patient)
        )

    optimizer = get_optimizer(model, training_args)
    scheduler = get_scheduler(optimizer, training_args, dataset)
    optimizers = (optimizer, scheduler)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.eval_dataset,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        callbacks=callbacks,
        optimizers=optimizers,
    )

    return trainer, dataset.predict_dataset

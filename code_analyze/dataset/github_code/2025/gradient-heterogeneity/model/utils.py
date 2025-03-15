from enum import Enum

from transformers import AutoConfig  # AutoModelForSequenceClassification,
from transformers import (
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
)

from model.model_with_linear_layer import RobertaForSequenceClassificationLinear


class TaskType(Enum):
    TOKEN_CLASSIFICATION = (1,)
    SEQUENCE_CLASSIFICATION = (2,)
    QUESTION_ANSWERING = (3,)
    MULTIPLE_CHOICE = 4


AUTO_MODELS = {
    TaskType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
    TaskType.SEQUENCE_CLASSIFICATION: RobertaForSequenceClassificationLinear,
    TaskType.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
    TaskType.MULTIPLE_CHOICE: AutoModelForMultipleChoice,
}


def get_model(
    model_args,
    task_type,
    config,
    fix_bert,
):
    if model_args.lora:
        raise ValueError("LoRA not supported")
    elif config.model_type == "roberta":
        if model_args.use_random_init:
            model = AUTO_MODELS[task_type](config)
        else:
            model = AUTO_MODELS[task_type].from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        bert_param = 0
    elif config.model_type == "roberta_random":
        model = AUTO_MODELS[task_type](config)
        bert_param = 0
    else:
        raise ValueError(f"Model type {config.model_type} not supported")
    if fix_bert:
        if config.model_type == "bert":
            for param in model.bert.parameters():
                param.requires_grad = False
            for _, param in model.bert.named_parameters():
                bert_param += param.numel()
        elif config.model_type == "roberta":
            for param in model.roberta.parameters():
                param.requires_grad = False
            for _, param in model.roberta.named_parameters():
                bert_param += param.numel()
        elif config.model_type == "deberta":
            for param in model.deberta.parameters():
                param.requires_grad = False
            for _, param in model.deberta.named_parameters():
                bert_param += param.numel()
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
    total_param = all_param - bert_param
    print("***** total param is {} *****".format(total_param))
    return model

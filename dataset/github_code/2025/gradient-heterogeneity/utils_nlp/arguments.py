from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.training_args
    """

    task_name: str = field(
        default="superglue",
        metadata={
            "help": "The name of the task to train on.",
        },
    )
    dataset_name: str = field(
        default="cb",
        metadata={
            "help": "The name of the dataset to use.",
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )
    template_id: Optional[int] = field(
        default=0, metadata={"help": "The specific prompt string to use"}
    )
    pilot: Optional[str] = field(
        default=None, metadata={"help": "do the pilot experiments."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="roberta-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    lora: bool = field(
        default=False, metadata={"help": "Will use lora during training"}
    )
    lora_r: int = field(default=8, metadata={"help": "The rank of lora"})
    lora_alpha: int = field(default=16, metadata={"help": "The length of prompt"})
    model_seed: int = field(
        default=1111, metadata={"help": "The random seed of model initialization."}
    )
    patient: int = field(
        default=10, metadata={"help": "The patient of early stopping."}
    )
    use_random_init: bool = field(
        default=False, metadata={"help": "Whether to use random init."}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    Custom TrainingArguments class to include new arguments.
    """

    # Training specification
    training_mode: str = field(
        default="normal",
        metadata={"help": "The training_mode method to use"},
    )
    no_early_stopping: bool = field(
        default=True, metadata={"help": "Whether to use early stopping or not"}
    )
    optimizer_name: str = field(
        default="adam", metadata={"help": "The optimizer to use"}
    )


def get_args_from_config(config):
    """Get args from config."""
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
    )

    args = parser.parse_dict(config)

    return args

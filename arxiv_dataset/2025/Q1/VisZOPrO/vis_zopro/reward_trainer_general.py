# this file contains a revision of TRL's rewardtrainer but for machine translation

import inspect
import warnings
from collections import defaultdict
from dataclasses import FrozenInstanceError, dataclass, replace
from typing import Any, Callable, Optional, Union

import pandas as pd
import torch
import torch.nn as nn
from accelerate.utils import gather_object
from datasets import Dataset
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from trl.data_utils import is_conversational
from trl import RewardConfig
from trl.trainer.utils import (
    compute_accuracy,
    decode_and_strip_padding,
    log_table_to_comet_experiment,
    print_rich_table,
)
from trl.trainer.reward_trainer import _tokenize


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb


# same as trl but accepts inputs with {"prompt", "chosen", "rejected", "completion"}
def apply_chat_template(
    example: dict[str, list[dict[str, str]]],
    tokenizer: "PreTrainedTokenizerBase",
    tools: Optional[list[Union[dict, Callable]]] = None,
) -> dict[str, str]:
    r"""
    Apply a chat template to a conversational example along with the schema for a list of functions in `tools`.

    For more details, see [`maybe_apply_chat_template`].
    """
    # Check that the example has the correct keys
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages", "label"]
    example_keys = {key for key in example.keys() if key in supported_keys}
    if example_keys not in [
        {"messages"},  # language modeling
        {"prompt"},  # prompt-only
        {"prompt", "completion"},  # prompt-completion
        {"prompt", "chosen", "rejected"},  # preference
        {"chosen", "rejected"},  # preference with implicit prompt
        {"prompt", "completion", "label"},  # unpaired preference
        {"prompt", "chosen", "rejected", "completion"},  # paired preference
    ]:
        raise KeyError(f"Invalid keys in the example: {example_keys}")

    # Apply the chat template to the whole conversation
    if "messages" in example:
        messages = tokenizer.apply_chat_template(example["messages"], tools=tools, tokenize=False)

    # Apply the chat template to the prompt, adding the generation prompt
    if "prompt" in example:
        prompt = tokenizer.apply_chat_template(
            example["prompt"], tools=tools, tokenize=False, add_generation_prompt=True
        )

    # Apply the chat template to the entire prompt + completion
    if "prompt" in example:  # explicit prompt and prompt-completion case
        if "chosen" in example:
            prompt_chosen = tokenizer.apply_chat_template(
                example["prompt"] + example["chosen"], tools=tools, tokenize=False
            )
            chosen = prompt_chosen[len(prompt) :]
        if "rejected" in example and "prompt" in example:  # explicit prompt
            prompt_rejected = tokenizer.apply_chat_template(
                example["prompt"] + example["rejected"], tools=tools, tokenize=False
            )
            rejected = prompt_rejected[len(prompt) :]
        if "completion" in example:
            prompt_completion = tokenizer.apply_chat_template(
                example["prompt"] + example["completion"], tools=tools, tokenize=False
            )
            completion = prompt_completion[len(prompt) :]
    else:  # implicit prompt case
        if "chosen" in example:
            chosen = tokenizer.apply_chat_template(example["chosen"], tools=tools, tokenize=False)
        if "rejected" in example:
            rejected = tokenizer.apply_chat_template(example["rejected"], tools=tools, tokenize=False)

    # Ensure that the prompt is the initial part of the prompt-completion string
    if "prompt" in example:
        error_message = (
            "The chat template applied to the prompt + completion does not start with the chat template applied to "
            "the prompt alone. This can indicate that the chat template is not supported by TRL."
            "\n**Prompt**:\n{}\n\n**Prompt + Completion**:\n{}"
        )
        if "chosen" in example and not prompt_chosen.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_chosen))
        if "rejected" in example and not prompt_rejected.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_rejected))
        if "completion" in example and not prompt_completion.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_completion))

    # Extract the completion by removing the prompt part from the prompt-completion string
    output = {}
    if "messages" in example:
        output["text"] = messages
    if "prompt" in example:
        output["prompt"] = prompt
    if "chosen" in example:
        output["chosen"] = chosen
    if "rejected" in example:
        output["rejected"] = rejected
    if "completion" in example:
        output["completion"] = completion
    if "label" in example:
        output["label"] = example["label"]

    return output


def tokenize_reward_general(batch: dict[str, list[Any]], tokenizer: "PreTrainedTokenizerBase") -> dict[str, list[Any]]:
    examples = {}
    if "score" in batch:
        examples = tokenize_reward(batch, tokenizer)
    if "chosen" in batch:
        ch_ex = _tokenize(batch, tokenizer)
        for key, value in ch_ex.items():
            if key not in examples:
                examples[key] = value
    return examples


def tokenize_reward(batch: dict[str, list[Any]], tokenizer: "PreTrainedTokenizerBase") -> dict[str, list[Any]]:
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
        # "score": [],
        # "prompt": [],
        # "completion": [],
    }

    for prompt, completion in zip(batch["prompt"], batch["completion"]):
        # Concatenate prompt and completion. Adjust the separator if needed.
        text = prompt + " " + completion
        tokenized = tokenizer(text)
        new_examples["input_ids"].append(tokenized["input_ids"])
        new_examples["attention_mask"].append(tokenized["attention_mask"])
        # new_examples["score"].append(example["score"])
        # # Keep the original text for visualisation.
        # new_examples["prompt"].append(example["prompt"])
        # new_examples["completion"].append(example["completion"])
    return new_examples


@dataclass
class RewardDataCollatorWithPaddingMT:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.

    Args:
        tokenizer (PreTrainedTokenizerBase):
            The tokenizer used for encoding the data.
        padding (Union[bool, str], optional, defaults to True):
            Padding strategy to pass to the tokenizer.
        pad_to_multiple_of (Optional[int], optional, defaults to None):
            If set, pads the sequence to a multiple of the provided value.
        return_tensors (str, optional, defaults to "pt"):
            The tensor type to use.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("No features provided.")

        batch = {}

        # Process ranking features if provided.
        if "input_ids_chosen" in features[0]:
            # Ensure all required ranking keys are present.
            required_ranking_keys = [
                "input_ids_chosen",
                "attention_mask_chosen",
                "input_ids_rejected",
                "attention_mask_rejected",
            ]
            missing_keys = [key for key in required_ranking_keys if key not in features[0]]
            if missing_keys:
                raise ValueError(
                    f"Ranking features must include {', '.join(required_ranking_keys)}. Missing: {', '.join(missing_keys)}"
                )

            features_chosen = []
            features_rejected = []
            margin_list = []
            has_margin = "margin" in features[0]

            for feature in features:
                for key in required_ranking_keys:
                    if key not in feature:
                        raise ValueError(
                            f"Each feature must include {', '.join(required_ranking_keys)}. Missing key: {key}"
                        )
                features_chosen.append({
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                })
                features_rejected.append({
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                })
                if has_margin:
                    margin_list.append(feature["margin"])

            batch_chosen = self.tokenizer.pad(
                features_chosen,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            batch_rejected = self.tokenizer.pad(
                features_rejected,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

            batch.update({
                "input_ids_chosen": batch_chosen["input_ids"],
                "attention_mask_chosen": batch_chosen["attention_mask"],
                "input_ids_rejected": batch_rejected["input_ids"],
                "attention_mask_rejected": batch_rejected["attention_mask"],
            })

            if has_margin:
                batch["margin"] = torch.tensor(margin_list, dtype=torch.float)

        # Process standard features if provided.
        if "input_ids" in features[0]:
            # Ensure required standard keys are present.
            if "attention_mask" not in features[0]:
                raise ValueError("Standard features must include `attention_mask`.")
            if "score" not in features[0]:
                raise ValueError("Standard features must include `score`.")

            tokenizable_features = [
                {"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask"]}
                for feature in features
            ]
            batch_standard = self.tokenizer.pad(
                tokenizable_features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            batch.update(batch_standard)
            scores = [feature["score"] for feature in features]
            batch["score"] = torch.tensor(scores, dtype=torch.float)

        if not batch:
            raise ValueError("Features do not have any expected keys for either ranking or standard tasks.")

        batch["return_loss"] = True
        return batch


# modification of TRL's RewardTrainer to work with machine translation as well as other tasks
class RewardTrainerGeneral(Trainer):
    _tag_names = ["trl", "reward-trainer"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        peft_config: Optional[dict] = None,
    ):
        """
        Modified RewardTrainer initialisation that now expects each training sample to have:
         - "prompt": the input prompt as a string
         - "completion": the model completion as a string
         - "score": a float (positive if good, negative if bad)
        The tokenisation step concatenates the prompt and completion and stores the resulting input IDs and attention mask.

        Also support normal RewardTrainer format
        """
        if max_length is not None and args.max_length is not None:
            raise ValueError(
                "You cannot specify both `max_length` and `args.max_length`. Please use the `RewardConfig` to set `max_length` once."
            )

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if not isinstance(model, PeftModel):
                if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_quantized", False):
                    _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
                        inspect.signature(prepare_model_for_kbit_training).parameters
                    )

                    prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                    if not _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        warnings.warn(
                            "You passed `gradient_checkpointing_kwargs` in the trainer's kwargs, but your peft version does not support it. "
                            "please update to the latest version of peft to use `gradient_checkpointing_kwargs`.",
                            UserWarning,
                        )
                    elif _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

                model = get_peft_model(model, peft_config)
        if compute_metrics is None:
            compute_metrics = compute_accuracy

        # When no data collator is provided, create one using the processing_class.
        if data_collator is None:
            if processing_class is None:
                raise ValueError(
                    "A processing_class must be specified when using the default RewardDataCollatorWithPadding"
                )
            if max_length is None:
                max_length = 512 if args.max_length is None else args.max_length

            data_collator = RewardDataCollatorWithPaddingMT(processing_class)

            # Warn user about remove_unused_columns if necessary.
            if args.remove_unused_columns:
                try:
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, set `remove_unused_columns=False` in your RewardConfig.",
                    UserWarning,
                )
            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False

        if "input_ids" not in train_dataset.column_names and "input_ids_chosen" not in train_dataset.column_names:
            fn_kwargs = {"tokenizer": processing_class}
            # drop unused columns, otherwise maybe_apply_chat_template will throw an error
            train_dataset = train_dataset.remove_columns(
                column_names=[col for col in train_dataset.column_names if col not in ["prompt", "completion", "score", "chosen", "rejected", "score"]]
            )
            if is_conversational(train_dataset[0]):
                train_dataset = train_dataset.map(apply_chat_template, fn_kwargs=fn_kwargs, load_from_cache_file=False)
            train_dataset = train_dataset.map(
                tokenize_reward_general,
                batched=True,
                fn_kwargs=fn_kwargs,
                num_proc=args.dataset_num_proc,
                load_from_cache_file=False,
            )

            # This filter is important because otherwise you get samples that exceed the model's context length and
            # get truncated => noisy signal the chosen/rejected label gets lost. The downside is that the
            # user might get surprised if N samples are missing from training.
            def filter_function(example):
                keep = True
                if "input_ids" in example:
                    keep = keep and len(example["input_ids"]) <= max_length
                if "input_ids_chosen" in example:
                    keep = keep and len(example["input_ids_chosen"]) <= max_length and len(example["input_ids_rejected"]) <= max_length
                return keep

            train_dataset = train_dataset.filter(
                filter_function,
                num_proc=args.dataset_num_proc,
            )
            if eval_dataset is not None:
                # drop unused columns
                eval_dataset = eval_dataset.remove_columns(
                    column_names=[col for col in eval_dataset.column_names if col not in ["prompt", "completion", "score", "chosen", "rejected"]]
                )
                if is_conversational(eval_dataset[0]):
                    eval_dataset = eval_dataset.map(apply_chat_template, fn_kwargs=fn_kwargs)
                eval_dataset = eval_dataset.map(
                    tokenize_reward_general,
                    batched=True,
                    fn_kwargs=fn_kwargs,
                    num_proc=args.dataset_num_proc,
                )
                # This filter is important because otherwise you get samples that exceed the model's context length and
                # get truncated => noisy signal the chosen/rejected label gets lost. The downside is that the
                # user might get surprised if N samples are missing from training.
                eval_dataset = eval_dataset.filter(
                    filter_function,
                    num_proc=args.dataset_num_proc,
                )
            else:
                args.eval_strategy = "no"

        # Flag to suppress token estimation warnings.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        # most tasks/dataset will only go trough one branch.
        # MT will go through both, in one branch it is learning the score of
        # one translation, in the other it is learning correct translations
        # REPRODUCIBILITY DETAIL: this was previously done through two different training loops (rather than combining the losses)

        total_loss = 0
        return_dict = {}
        if "input_ids" in inputs:
            """
            Computes the loss by having the model predict a scalar reward for the concatenated (prompt+completion)
            and comparing it to the provided score using mean squared error.
            """
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
            # Assume the model returns logits of shape (batch, 1)
            predicted_reward = outputs["logits"]
            loss = torch.nn.functional.mse_loss(predicted_reward.squeeze(1), inputs["score"])
            total_loss += loss
            if return_outputs:
                return_dict["predicted_reward"] = predicted_reward
        if "input_ids_chosen" in inputs:
            rewards_chosen = model(
                input_ids=inputs["input_ids_chosen"],
                attention_mask=inputs["attention_mask_chosen"],
                return_dict=True,
            )["logits"]
            rewards_rejected = model(
                input_ids=inputs["input_ids_rejected"],
                attention_mask=inputs["attention_mask_rejected"],
                return_dict=True,
            )["logits"]
            # calculate loss, optionally modulate with margin
            if "margin" in inputs:
                loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
            else:
                loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

            if self.args.center_rewards_coefficient is not None:
                loss += self.args.center_rewards_coefficient * torch.mean((rewards_chosen + rewards_rejected) ** 2)
            total_loss += loss
            if return_outputs:
                return_dict["rewards_chosen"] = rewards_chosen
                return_dict["rewards_rejected"] = rewards_rejected
        if return_outputs:
            return total_loss, return_dict
        return total_loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)
        # Stack accepted against rejected, mean over logits
        # and softmax to get preferences between accepted and rejected to sum to 1
        logits = torch.stack(logits).mean(dim=2).softmax(dim=0).T

        labels = torch.zeros(logits.shape[0])
        labels = self._prepare_inputs(labels)

        return loss, logits, labels

    def evaluate(self, *args, **kwargs):
        num_print_samples = kwargs.pop("num_print_samples", 4)
        self.visualize_samples(num_print_samples)
        return super().evaluate(*args, **kwargs)

    def visualize_samples(self, num_print_samples: int):
        """
        Visualise some samples by printing the prompt, completion, the predicted reward, and the actual score.
        """
        if self.eval_dataset is None:
            return
        eval_dataloader = self.get_eval_dataloader()
        table = defaultdict(list)
        for _, inputs in enumerate(eval_dataloader):
            _, logits, _ = self.prediction_step(self.model, inputs, prediction_loss_only=False)
            if "input_ids" in inputs:
                prompt_text = inputs.get("prompt", decode_and_strip_padding(inputs["input_ids"], self.processing_class))
                completion_text = inputs.get("completion", "")
                table["prompt"].extend(gather_object(prompt_text))
                table["completion"].extend(gather_object(completion_text))
                # Round the predicted rewards for readability.
                table["score"].extend(gather_object(inputs["score"]))
            else:
                chosen_text = decode_and_strip_padding(inputs["input_ids_chosen"], self.processing_class)
                rejected_text = decode_and_strip_padding(inputs["input_ids_rejected"], self.processing_class)
                table["chosen_text"].extend(gather_object(chosen_text))
                table["rejected_text"].extend(gather_object(rejected_text))
            table["logits"].extend(
                gather_object([[round(inner_item, 4) for inner_item in item] for item in logits.tolist()])
            )
            if num_print_samples >= 0 and len(table["prompt"]) >= num_print_samples:
                break
        df = pd.DataFrame(table)
        if self.accelerator.process_index == 0:
            print_rich_table(df[:num_print_samples])
            if "wandb" in self.args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in self.args.report_to:
                log_table_to_comet_experiment(name="completions.csv", table=df)

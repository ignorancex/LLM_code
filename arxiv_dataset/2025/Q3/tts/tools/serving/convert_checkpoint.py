"""A script to convert a checkpoint to a serving-friendly format.

Example usage:
python tools/serving/convert_checkpoint.py \
  --checkpoint_path=./checkpoints/tts-model-epoch-100.pt \
  --output_path=./checkpoints/serving/tts-model-epoch-100
"""

import json
import math
import os
import time
from typing import Tuple

import transformers
from absl import app, flags, logging

from tts.core import constants, modeling

FLAGS = flags.FLAGS

_CHECKPOINT_PATH = flags.DEFINE_string("checkpoint_path", None,
                                       "Path to the checkpoint file.")
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", None, "Path to save the converted checkpoint. If not provided, the "
    "checkpoint will be saved to where --checkpoint_path points to.")
_MERGE_LORA = flags.DEFINE_bool(
    "merge_lora", False,
    "Whether to merge the LoRA weights into the base model (if it is a LoRA model).")
_UPDATE_EOS_TOKEN = flags.DEFINE_bool(
    "update_eos_token", True,
    "Whether to update the eos token to the speech end token.")
_ADD_NONVERBAL_TOKENS = flags.DEFINE_bool("add_missing_nonverbal_tokens", False,
                                          "Whether to add missing nonverbal tokens.")


def _round_vocab_size(size: int, multiple: int = 64) -> int:
    """Round up the vocabulary size to the nearest multiple."""
    return math.ceil(size / multiple) * multiple


def _add_nonverbal_tokens_and_pad(
    tokenizer: transformers.PreTrainedTokenizerBase, model: transformers.PreTrainedModel
) -> Tuple[transformers.PreTrainedTokenizerBase, transformers.PreTrainedModel]:
    """Processes the tokenizer and model to add missing nonverbal tokens."""
    existing_vocab = tokenizer.get_vocab()
    missing_tokens = [t for t in constants.NONVERBAL_TOKENS if t not in existing_vocab]
    if missing_tokens:
        tokenizer.add_tokens(missing_tokens)
        logging.info("Added %d missing nonverbal tokens: %s", len(missing_tokens),
                     missing_tokens)
    else:
        logging.info("All nonverbal tokens already present in tokenizer.")

    # Round up vocab size to avoid performance issues.
    current_vocab_size = len(tokenizer)
    target_vocab_size = _round_vocab_size(current_vocab_size)
    if target_vocab_size > current_vocab_size:
        padding_tokens = [
            f"<|pad_token_{i}|>" for i in range(current_vocab_size, target_vocab_size)
        ]
        tokenizer.add_tokens(padding_tokens)
        logging.info("Padded vocab to %d with %d extra tokens.", target_vocab_size,
                     len(padding_tokens))

    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def _update_eos_token(
    tokenizer: transformers.PreTrainedTokenizerBase, model: transformers.PreTrainedModel
) -> Tuple[transformers.PreTrainedTokenizerBase, transformers.PreTrainedModel]:
    """Updates the EOS token in the tokenizer and model to the speech end token."""
    tokenizer.eos_token = constants.SPEECH_END_TOKEN
    eos_token_id = tokenizer.convert_tokens_to_ids(constants.SPEECH_END_TOKEN)
    model.generation_config.eos_token_id = eos_token_id
    return tokenizer, model


def main(argv: list[str]) -> None:
    del argv  # Unused.

    checkpoint_path = _CHECKPOINT_PATH.value
    checkpoint_dir = os.path.dirname(checkpoint_path)
    output_path = checkpoint_dir if _OUTPUT_PATH.value is None else _OUTPUT_PATH.value
    logging.info("Converting checkpoint from [%s] to [%s]...", checkpoint_path,
                 output_path)

    tokenizer, config, model, lora_config = modeling.load_tokenizer_config_and_model(
        checkpoint_path)

    if _MERGE_LORA.value:
        if lora_config is None:
            raise ValueError("LoRA config is not found in the config file.")
        model = model.merge_and_unload()
        logging.info("LoRA weights merged into the base model.")

    if _ADD_NONVERBAL_TOKENS.value:
        tokenizer, model = _add_nonverbal_tokens_and_pad(tokenizer, model)

    if _UPDATE_EOS_TOKEN.value:
        tokenizer, model = _update_eos_token(tokenizer, model)

    start_time = time.time()
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)
    with open(os.path.join(output_path, constants.CONFIG_FILE_NAME), "w") as f:
        json.dump(config, f, indent=4)
    logging.info("Model saved to [%s] in %.2f seconds.", output_path,
                 time.time() - start_time)


if __name__ == "__main__":
    flags.mark_flag_as_required("checkpoint_path")
    app.run(main)

import gc
import random
from typing import List, Optional, Tuple

import numpy
import torch
from absl import flags
from transformers.models.bart.modeling_bart import shift_tokens_right

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_type", "all_finetune", "The type of experiment with the LM model.")

flags.DEFINE_integer("prompt_length", 20, "length of the prompts in the input sequence.")


def prepend_prompt(
    input_ids: torch.LongTensor, mask: torch.LongTensor, prompt_tokens: List[int], lm_type: Optional[str] = "roberta"
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Prepend the input_ids with the prompt token ids after the first BOS
    token.

    - input_ids and mask are the raw inputs to t5 to be modified.
    """
    batch_size, sequence_length = input_ids.size()

    prompt_tensor = torch.tensor(prompt_tokens, device=input_ids.device)

    prompt_tensor = prompt_tensor.view(1, FLAGS.prompt_length).expand(batch_size, FLAGS.prompt_length)

    # prompt tokens are always valid so mask is always 1.
    prompt_mask = torch.ones((batch_size, FLAGS.prompt_length), device=mask.device)

    if lm_type in ["roberta", "llama2"]:
        # put prompt tokens after the first BOS token.
        prompted_input_ids = torch.cat((input_ids[:, 0].view(batch_size, 1), prompt_tensor, input_ids[:, 1:]), dim=1)
    elif lm_type == "t5":
        # put prompt tokens before the input tokens.
        prompted_input_ids = torch.cat((prompt_tensor, input_ids), dim=1)

    # the mask on the BOS token is always 1.
    prompted_mask = torch.cat((prompt_mask, mask), dim=1)
    return prompted_input_ids, prompted_mask


def modify_inputs(
    batch: torch.utils.data.Dataset, prompt_lists: Optional[List[List[int]]] = None, lm_type: Optional[str] = "roberta"
) -> None:
    """This function will modify the input_ids and mask in the batch to include
    prompt tokens if needed.

    If we have multiple prompt tokens to augment the input with, it will
    repeat the outputs per prompt template.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    modified_input_ids_name = "modified_input_ids"
    modified_attention_mask_name = "modified_attention_mask"

    batch_size, sequence_length = input_ids.size()
    batch[modified_input_ids_name] = input_ids
    batch[modified_attention_mask_name] = attention_mask
    if FLAGS.exp_type == "soft_prompt_finetune":
        # This is used for soft prompt tuning! Then for a prompt with length |P|,
        # we add dummy prompt token ids from [0, |P|-1] to map
        # those into |P| vectors from the prompt embedder.
        (batch[modified_input_ids_name], batch[modified_attention_mask_name]) = prepend_prompt(
            input_ids, attention_mask, prompt_tokens=list(range(FLAGS.prompt_length)), lm_type=lm_type
        )

    elif FLAGS.exp_type in ["gradient_search", "grips"] and prompt_lists:
        input_ids_stack = []
        input_mask_stack = []
        num_prompts = 0
        for prompt_tokens in prompt_lists:
            input_ids_per_template, mask_per_template = prepend_prompt(
                input_ids, attention_mask, prompt_tokens, lm_type=lm_type
            )
            input_ids_stack.append(input_ids_per_template)
            input_mask_stack.append(mask_per_template)
            num_prompts += 1

        batch[modified_input_ids_name] = torch.stack(input_ids_stack, dim=1).view(num_prompts * batch_size, -1)
        batch[modified_attention_mask_name] = torch.stack(input_mask_stack, dim=1).view(num_prompts * batch_size, -1)


def log_of_labels(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    input_mask: torch.Tensor,
    decoder_mask: torch.Tensor,
    labels: torch.Tensor,
    loss_func: torch.nn.CrossEntropyLoss,
) -> torch.Tensor:
    """Do a forward computation and compute the log probability for the given
    labels."""

    # shift the gold labels one step to the right and do teacher-forcing by giving the gold previous token
    # and then compute the probablity for the next token at each step.
    # labels = [pos, it, ive]
    # decoder_input = [<BOS>, pos, it]
    # we want to see what is log probability for the target sequence "positive".
    output = model(
        input_ids=input_ids,
        attention_mask=input_mask,
        decoder_attention_mask=decoder_mask,
        decoder_input_ids=shift_tokens_right(labels, model.config.pad_token_id, model.config.decoder_start_token_id),
        labels=None,
    )

    log_p = -loss_func(
        output.logits.view(-1, output.logits.size(-1)),
        labels.view(-1),
    )
    batch_size, sequence_length, vocab_size = output.logits.size()
    # compute per-token log probability in a sequence.
    # log_p has log probabilities for the following target output: [pos, it, ive]
    log_p = log_p.view(batch_size, sequence_length)

    # pad tokens have index -100 in huggingface.
    good_log_p = log_p.masked_fill_(labels == -100, 0.0)

    # good_log_p now has the log probability of the output sequence tokens.
    # sum over the sequence length to compute the log probability for a full sequence.
    return torch.sum(good_log_p, dim=1).squeeze()


def mlm_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    input_mask: torch.Tensor,
) -> torch.Tensor:
    """Do a forward computation and compute the logits for the given
    input_ids for a masked language model such as roberta!"""

    output = model(
        input_ids=input_ids,
        attention_mask=input_mask,
        labels=None,
    )
    return output.logits


def mlm_log_of_labels(
    logits: torch.Tensor, labels: torch.Tensor, loss_func: torch.nn.CrossEntropyLoss
) -> torch.Tensor:
    """Compute the actual log of labels given pre-computed logits."""

    log_p = -loss_func(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )

    batch_size, sequence_length, vocab_size = logits.size()

    # compute per-token log probability in a sequence.
    log_p = log_p.view(batch_size, sequence_length)

    # non-masked tokens have index -100 in huggingface.
    good_log_p = log_p.masked_fill_(labels == -100, 0.0)

    # good_log_p now has the log probability of the output
    # sequence tokens corresponding to the labels at the [MASK] location.
    return torch.sum(good_log_p, dim=1)


def clear_cache() -> None:
    """Clean unused GPU Cache!"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def set_random_seed(seed: int) -> None:
    """Set the random seed, which initializes the random number generator.

    Ensures that runs are reproducible and eliminates differences due to
    randomness.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def z_scoring(rewards: torch.FloatTensor) -> torch.FloatTensor:
    """Perform normalization of the rewards."""
    rewards_mean = torch.mean(rewards, dim=1, keepdim=True)
    rewards_std = torch.std(rewards, dim=1, keepdim=True)
    return (rewards - rewards_mean) / (rewards_std + 1e-12)

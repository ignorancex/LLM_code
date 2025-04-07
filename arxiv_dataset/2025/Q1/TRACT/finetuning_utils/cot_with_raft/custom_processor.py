from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from llamafactory.extras import logging
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.data.processors.supervised import _encode_supervised_example


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from llamafactory.hparams import DataArguments
    from llamafactory.data.template import Template


logger = logging.get_logger(__name__)

# This function is modified from 

# This function is modified from https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/processors/supervised.py


def preprocess_cot_raft_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        input_ids, original_labels = _encode_supervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )

        # TODO: I think the sequence is not yet padded here. Verify this

        # Step 1: Use the attention mask to calculate the sequence length
        # logger.info(f"\n\nSequence length: {seq_len}")
        # logger.info(f"The input ids correspond to the following tokens: {tokenizer.convert_ids_to_tokens(input_ids)}\n\n")
        # Step 2: Construct another score_label

        # The following is customized for Mistral-7B-Instruct-v0.2
        indices_to_scores = {
            28740: 1.0,
            28750: 2.0,
            28770: 3.0,
            28781: 4.0,
            28782: 5.0,
        }

        # The following is customized for LLama-3.1-8B-Instruct
        #  indices_to_scores = {
        #      16: 1.0,
        #      17: 2.0,
        #      18: 3.0,
        #      19: 4.0,
        #      20: 5.0,
        #  }
        
        possible_scores = tokenizer.convert_ids_to_tokens(list(indices_to_scores.keys()))
        possible_scores = [int(score) for score in possible_scores]
        for idx, score in enumerate(possible_scores):
            if score != idx + 1:
                raise ValueError(f"Indices and scores do not match: {possible_scores} and {list(indices_to_scores.keys())}")
        score_label = indices_to_scores[original_labels[-2]]

        # Step 3: Mask out the second-to-last token in the input_ids. This token corresponds to the score token, whose loss will be calculated using raft
        lm_loss_labels = original_labels.copy()
        lm_loss_labels[-2] = IGNORE_INDEX

         
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(lm_loss_labels)
        model_inputs["score_labels"].append(score_label)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs
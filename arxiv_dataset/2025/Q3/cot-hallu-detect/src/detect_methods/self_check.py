
from typing import Literal
import torch
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from utils import time_performance_decorator

# see: https://github.com/potsawee/selfcheckgpt/

SELF_CHECK_PROMPT = """Context: {}\n\nSentence: {}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer:"""


@time_performance_decorator(enable=False)
@torch.no_grad()
def self_check_nli(
    answer: str,
    samples: list[str],
    nli_model: DebertaV2ForSequenceClassification,
    nli_tokenizer: DebertaV2Tokenizer,
):

    inputs = nli_tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=[
            (answer, sample) for sample in samples
        ],
        add_special_tokens=True, padding="longest",
        truncation=True, return_tensors="pt",
        return_token_type_ids=True, return_attention_mask=True,
    )
    inputs = inputs.to(nli_model.device)
    logits = nli_model(**inputs).logits  # neutral is already removed
    probs = torch.softmax(logits, dim=-1)
    prob_ = probs[:, 1]  # prob(contradiction)

    return prob_.mean(dim=-1).item()


@time_performance_decorator(enable=False)
@torch.no_grad()
def self_check_prompt(
    llm: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    answer: str,
    samples: list[str],
):
    option_ids: torch.Tensor = tokenizer.encode(
        " Yes No",
        add_special_tokens=False,
        return_tensors="pt",
    ).to(llm.device)

    # this seems to improve performance when using the simple prompt template
    samples = [sample.replace("\n", "").strip() for sample in samples]
    prompts = [
        SELF_CHECK_PROMPT.format(sample, answer)
        for sample in samples
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(llm.device)
    outputs = llm.forward(**inputs)
    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    option_ids = option_ids.repeat(len(prompts), 1)
    probs_ = (
        probs
        .gather(index=option_ids, dim=-1)
        .to(device="cpu", dtype=torch.float64)
        .numpy()
    )
    scores = probs_[:, 0] / (probs_[:, 0] + probs_[:, 1] + 1e-7)
    score = scores.mean().item()
    return score


@time_performance_decorator(enable=False)
@torch.no_grad()
def calcu_self_check_nli_and_prompt(
    llm: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    nli_model: DebertaV2ForSequenceClassification,
    nli_tokenizer: DebertaV2Tokenizer,
    samples: list[str],
    answer: str,
    method: Literal["all", "self_check_nli", "self_check_prompt"] = "all",
):
    nli_score, prompt_score = 0, 0
    if method in ("self_check_nli", "all"):
        nli_score = self_check_nli(answer, samples, nli_model, nli_tokenizer)
    elif method in ("self_check_prompt", "all"):
        prompt_score = self_check_prompt(llm, tokenizer, answer, samples)
    return nli_score, prompt_score

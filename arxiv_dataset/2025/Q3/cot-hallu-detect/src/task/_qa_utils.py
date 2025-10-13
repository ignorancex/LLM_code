
import re
import string
from typing import Callable

import torch
from transformers import (
    PreTrainedTokenizerBase,
    RobertaModel, RobertaTokenizer
)


# Official evaluation script for v1.0 of the TriviaQA dataset.
# Extended from the evaluation script for v1.1 of the SQuAD dataset.


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def calcu_exact_match_score(prediction: str, ground_truth: str):
    if len(prediction) == 0:
        return 0.
    if (
        (normalize_answer(prediction) == normalize_answer(ground_truth)) or
        (normalize_answer(ground_truth) in normalize_answer(prediction)) or
        (normalize_answer(prediction) in normalize_answer(ground_truth))
    ):
        return 1.
    else:
        return 0.


def _mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging
    see: https://huggingface.co/sentence-transformers/nli-roberta-large"""
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@torch.no_grad()
def calcu_sentences_sims(
    prediction: str, ground_truths: list[str], question: str,
    embd_model: RobertaModel, embd_tokenizer: RobertaTokenizer,
):
    sentences = [
        prediction,
        "\n".join(ground_truths),
        f"{question}\n{'\n'.join(ground_truths)}"
    ] + [
        f"{question}\n{gt}" for gt in ground_truths
    ] + ground_truths
    encoded_input = embd_tokenizer(
        sentences, padding=True,
        truncation=True, return_tensors='pt',
    )
    model_output = embd_model(**encoded_input)
    sentence_embeddings = _mean_pooling(
        model_output, encoded_input['attention_mask'])
    tgt = sentence_embeddings[1:]
    src: torch.Tensor = sentence_embeddings[0].unsqueeze(0)
    sims = torch.cosine_similarity(
        src.repeat_interleave(tgt.shape[0], dim=0),
        tgt, dim=1
    ).to("cpu").numpy().tolist()
    return max(sims)


def _longest_common_subseq(seq1: list, seq2: list) -> int:
    dp = [[0] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(seq1)][len(seq2)]


def calcu_rouge_l(
    prediction: str,
    ground_truth: str,
    tokenizer: PreTrainedTokenizerBase
) -> dict[str, float]:

    pred_tokens = tokenizer.tokenize(normalize_answer(prediction))
    gt_tokens = tokenizer.tokenize(normalize_answer(ground_truth))

    lcs_length = _longest_common_subseq(pred_tokens, gt_tokens)

    precision = lcs_length / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = lcs_length / len(gt_tokens) if len(gt_tokens) > 0 else 0.0
    f1 = (
        (2 * precision * recall) /
        (precision + recall) if (precision + recall) > 0 else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def calcu_max_rouge_l_over_ground_truths(
    prediction: str,
    ground_truths: list[str],
    tokenizer: PreTrainedTokenizerBase
):
    res_all = [
        calcu_rouge_l(prediction, text_gt, tokenizer)
        for text_gt in ground_truths
    ]
    return {
        "precision": max([res["precision"] for res in res_all]),
        "recall": max([res["recall"] for res in res_all]),
        "f1": max([res["f1"] for res in res_all])
    }


def metric_max_over_ground_truths(
    metric_fn: Callable[[str, str], float],
    prediction: str,
    ground_truths: list[str],
):
    return max(
        metric_fn(prediction, ground_truth)
        for ground_truth in ground_truths
    )

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm
import torch
from t5.evaluation import metrics as mt
from transformers import PreTrainedModel, PreTrainedTokenizer, DataCollatorForSeq2Seq
import json
import numpy as np
from datasets import Dataset


logger = logging.getLogger(__name__)

# Dictionary for storing metrics so that we can easily get them from the
# prompt templates
METRICS_DICT = {
    "BLEU"                : mt.bleu,
    "ROUGE"               : mt.rouge,
    "Span Squad"          : mt.span_squad,
    "Squad"               : mt.squad,
    "Trivia QA"           : mt.trivia_qa,
    "Accuracy"            : mt.accuracy,
    "Sequence Accuracy"   : mt.sequence_accuracy,
    "Pearson Correlation" : mt.pearson_corrcoef,
    "Spearman Correlation": mt.spearman_corrcoef,
    "MultiRC"             : mt.multirc_f1_over_all_answers,
    "AUC"                 : mt.auc,
    "COQA F1"             : mt.coqa_f1,
    "Edit Distance"       : mt.edit_distance,
    "Other"               : mt.accuracy,
}


def serialize_prediction(
        idx,
        prediction,
        target,
        input_seq,
        choice_logits=None
) -> str:
    """
    Helper function to turn predictions into writeable string.

    Args:
        idx (int): id of the prediction
        prediction (list): List of prediction strings.
        target (str): The target string.
        input_seq (str): The input string.
        choice_logits (dict): The choices (if exist) dict with log probabilities.

    """
    return json.dumps({
        "id"           : idx,
        "prediction"   : prediction,
        "target"       : target,
        "input"        : input_seq,
        "choice_logits": choice_logits or {}
    })


def score_choice(logits, choice, length_normalize: bool = False):
    return torch.sum(logits[:, :, choice], (1, 2)) / (1 if not length_normalize else len(choice))


def generate_predictions_choices(
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        device: torch.device,
        length_normalize: bool = False,
        batch_size: int = 1
) -> Dict[str, List]:
    """
    Generate predictions when using answer choices. It WILL NOT handle batch
    sizes of more than 1 for the time being.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader to use.
        choices_tokenized (List[List[int]]): The choices tokenized.
        model (PreTrainedModel: Model to use.
        device (torch.device): device to use.
        length_normalize (bool): Use length normalization

    Returns:
        predictions (Dict[str, List]): Dict with two keys "targets" and
            "scores". "targets" is a list of tuples with values:

            `((example idx, choice idx), is the choice correct, weight)`

            The list of "scores" is the log probability of the choice.

            NOTE: The number of elements in both is the number of choices *
            number of examples.
    """
    logger.info(f"Generating Choices")
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=2,
        max_length=1024,
        padding='longest',
        label_pad_token_id=tokenizer.pad_token_id
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False
    )

    targets = []
    dataset_scores = []
    with torch.no_grad():
        batch_num = 0
        for batch in tqdm(data_loader, desc="Generating"):
            generated = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            logits = generated.logits.cpu().detach()
            for i in range(logits.shape[0]):
                targets.append(
                    (batch['idx'][i].tolist(), batch['is_correct'][i].tolist(), 1.0)
                )

            choice_mask = batch['labels_attention_mask']
            choice_mask[torch.arange(logits.shape[0]), batch['labels_len'] - 1] = 0
            choice_logits = torch.gather(
                logits, -1, batch['labels'].unsqueeze(-1)
            ).squeeze(-1) * choice_mask
            scores = choice_logits.sum(-1)

            if length_normalize:
                scores /= choice_mask.sum(-1)

            dataset_scores.extend(scores.tolist())

            if (batch_num + 1) % 10 == 0:
                torch.cuda.empty_cache()

    return {'targets': targets, "scores": dataset_scores}


def evaluate(
        predictions: Dict[str, List],
        choices: List[str],
        source_dataset: Dataset,
        tokenized_dataset: Dataset,
        out_path: Path
):
    # This program is only made for fixed choice tasks. So we can assume all #
    # of choices will stay the same.
    num_choices = len(choices)

    aligned_preds = [[None] * num_choices for _ in range(len(source_dataset))]
    aligned_targets = [None] * len(source_dataset)

    # Align the predictions with the targets and the source data so that they
    # can be saved to a file. Also save them to their own arrays so that we can
    # calculate per class F1
    for target, score in zip(predictions['targets'], predictions['scores']):
        (idx, choice_idx), is_correct, _ = target
        if is_correct:
            aligned_targets[idx] = choice_idx
        aligned_preds[idx][choice_idx] = score

    aligned_preds = np.array(aligned_preds)
    predicted_choice = aligned_preds.argmax(-1)

    final_metrics = mt.rank_classification(
        predictions['targets'],
        predictions['scores'],
        num_classes=num_choices
    )

    # By default, rank classification returns mean_{num_classes}_f1. Easier to
    # handle across tasks if we rename it to F1. In the case of 2 choices, it
    # is only called F1 so no change.
    try:
        final_metrics['f1'] = final_metrics.pop(f"mean_{num_choices}class_f1")
    except KeyError:
        assert 'f1' in final_metrics
    f1_metrics = mt.sklearn_metrics_wrapper(
        "fbeta_score",
        metric_dict_str="f1_by_class",
        metric_post_process_fn=lambda x: 100 * x,
        beta=1,
        labels=range(num_choices),
        average=None
    )(
        predicted_choice, np.array(aligned_targets)
    )

    for l, f1 in enumerate(f1_metrics['f1_by_class']):
        final_metrics[f"f1_choice_{l + 1}"] = f1
    input_lens = tokenized_dataset['input_len']
    final_metrics['input_len/mean'] = np.mean(input_lens)  # type: ignore
    final_metrics['input_len/std'] = np.std(input_lens)  # type: ignore

    sorted_scores = np.sort(aligned_preds, axis=-1)
    scores_ptp = np.abs(np.ptp(sorted_scores, -1))
    diff_places = np.abs(sorted_scores[:, :-1] - sorted_scores[:, 1:])
    ranks = np.argsort(-aligned_preds) + 1

    final_metrics['logits/range_mean'] = np.mean(scores_ptp)  # type: ignore
    final_metrics['logits/range_std'] = np.std(scores_ptp)  # type: ignore

    for i in range(len(choices)):
        if i < len(choices) - 1:
            diff_places_for_i = diff_places[:, i]
            met_key = f'logits/diff_{i + 1}_to_{i + 2}'
            final_metrics[f'{met_key}_mean'] = np.mean(diff_places_for_i)  # type: ignore
            final_metrics[f'{met_key}_std'] = np.std(diff_places_for_i)  # type: ignore

        ranks_of_choice_i = ranks[:, i]
        met_key = f"logits/choice_{i + 1}_rank"
        final_metrics[f'{met_key}_mean'] = np.mean(ranks_of_choice_i)  # type:ignore
        final_metrics[f'{met_key}_std'] = np.std(ranks_of_choice_i)  # type: ignore

    for k, v in final_metrics.items():
        logger.info(f"{k:>32}: {v:.2f}")

    pred_file = out_path.joinpath('predictions.jsonl')
    with pred_file.open('w', encoding='utf-8') as pred_fp:
        for i, preds in enumerate(aligned_preds):
            ex = source_dataset[i]
            serialized = serialize_prediction(
                idx=i,
                prediction=choices[predicted_choice[i]],
                target=ex['output'],
                input_seq=ex['prompt'],
                choice_logits={
                    i: logit
                    for i, logit in enumerate(aligned_preds[i])
                }
            )
            pred_fp.write(serialized + '\n')

    metrics_file = out_path.joinpath("metrics.json")
    logger.info(f"Saving metrics to '{metrics_file}'")
    with metrics_file.open('w', encoding='utf-8') as fp:
        fp.write(json.dumps(final_metrics, indent=True))

    return metrics_file, pred_file

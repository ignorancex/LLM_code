import json
import math

import pytest
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, T5Config
from datasets import load_dataset
from pathlib import Path
from unittest.mock import MagicMock
import torch
import numpy as np
from promptsource.templates import DatasetTemplates, Template
from t5.evaluation import metrics as mt
import os

from src import evaluation
from src.preprocessing import preprocess_dataset


@pytest.mark.parametrize("length_normalized", [True, False],
                         ids=["W/Normalization", "No Normalization"])
@pytest.mark.parametrize("device_name", ['cpu', 0], ids=["CPU", "GPU"])
@pytest.mark.parametrize("batch_size", [1, 2], ids=["Single", "Double"])
def test_generate_prediction_choices(length_normalized, device_name, batch_size):
    prompt_name = "guaranteed true"
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    original_dataset = load_dataset("anli", split="train_r1[:16]")
    prompt: Template = DatasetTemplates('anli')[prompt_name]
    device = torch.device(device_name)
    # batch_size = 1
    tokenized, ds, choices_tokenized = preprocess_dataset(
        "Testing",
        original_dataset,
        tokenizer,
        prompt,
        num_proc=1
    )

    expected_targets = {}
    expected_scores = {}

    with torch.no_grad():
        model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
        model.eval()
        for batch in tokenized.to_dict(batched=True, batch_size=batch_size):
            padded = tokenizer.pad(
                {'input_ids': batch['input_ids'], "attention_mask": batch['attention_mask']},
                padding="longest",
                max_length=1024,
            )

            model_output = model(
                input_ids=torch.tensor(padded['input_ids'], device=device),
                attention_mask=torch.tensor(padded['attention_mask'], device=device),
                labels=torch.tensor(batch['labels'], device=device)
            )
            logits = model_output.logits.cpu().detach()
            for b in range(batch_size):
                item_key = f"{batch['idx'][b][0]}|{batch['idx'][b][1]}"

                expected_targets[item_key] = (
                    (batch['idx'][b], batch['is_correct'][b], 1.0)
                )
                choice_idx = batch['choice_idx'][b]

                score = 0
                gathered_logits = []
                for i, t in enumerate(choices_tokenized[choice_idx]):
                    score += logits[b, i, t].item()
                    gathered_logits.append(logits[b, i, t].item())
                expected_scores[item_key] = score / (
                    1 if not length_normalized else len(choices_tokenized[choice_idx])
                )

    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
    model.eval()
    result = evaluation.generate_predictions_choices(
        tokenized.sort('choice_idx'),
        tokenizer=tokenizer,
        model=model,
        device=device,
        length_normalize=length_normalized, batch_size=batch_size
    )

    assert len(result['targets']) == len(expected_targets)
    assert len(result['scores']) == len(expected_scores)

    for i, (target, score) in enumerate(zip(result['targets'], result['scores'])):
        idx_pair, _, _ = target
        item_key = f"{idx_pair[0]}|{idx_pair[1]}"
        assert item_key in expected_targets, f"{i} missing {item_key}"
        assert item_key in expected_scores, f"{i} missing {item_key}"
        expected_target = expected_targets.pop(item_key)
        expected_score = expected_scores.pop(item_key)

        assert target == expected_target, f"{i}:{item_key} does not have correct target"
        assert math.isclose(
            score, expected_score, rel_tol=1e-3
        ), f"{i}:{item_key} does not have correct score"

    assert set(expected_targets) == set()
    assert set(expected_scores) == set()


def test_evaluate(tmpdir):
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    original_dataset = load_dataset("anli", split="train_r1[:2]")
    prompt: Template = DatasetTemplates('anli')["can we infer"]
    choices = prompt.get_fixed_answer_choices_list()
    num_choices = len(choices)

    tokenized, ds, _ = preprocess_dataset(
        "Testing",
        original_dataset,
        tokenizer,
        prompt,
        num_proc=1
    )
    predictions = {
        "targets": [([1, 2], False, 1.0), ([1, 1], True, 1.0), ([1, 0], False, 1.0),
                    ([0, 2], False, 1.0), ([0, 1], False, 1.0), ([0, 0], True, 1.0)],
        "scores" : [-55.96, -88.09, -32.92, -55.22, -89.0, -32.75]
    }

    pred_scores = [
        [-32.75, -89.0, -55.22],
        [-32.92, -88.09, -55.96]
    ]
    targets = [1, 0]
    preds = np.array(pred_scores).argmax(-1)

    sorted_scores = np.sort(pred_scores, axis=-1)
    scores_ptp = np.abs(np.ptp(sorted_scores, -1))
    diff_places = np.abs(sorted_scores[:, :-1] - sorted_scores[:, 1:])
    ranks = np.argsort(-np.array(pred_scores)) + 1

    expected_metrics = mt.rank_classification(
        predictions['targets'],
        predictions['scores'],
        num_classes=num_choices
    )
    f1_fn = mt.sklearn_metrics_wrapper(
        "fbeta_score",
        metric_dict_str="f1_by_class",
        metric_post_process_fn=lambda x: 100 * x,
        beta=1,
        labels=range(num_choices),
        average=None
    )
    f1_metrics = f1_fn(
        preds, np.array(targets)
    )
    expected_metrics['f1'] = expected_metrics.pop('mean_3class_f1')

    expected_metrics["f1_choice_1"] = f1_metrics['f1_by_class'][0]
    expected_metrics["f1_choice_2"] = f1_metrics['f1_by_class'][1]
    expected_metrics["f1_choice_3"] = f1_metrics['f1_by_class'][2]
    expected_metrics['input_len/mean'] = np.mean(tokenized['input_len'])  # type: ignore
    expected_metrics['input_len/std'] = np.std(tokenized['input_len'])  # type: ignore
    expected_metrics['logits/range_mean'] = np.mean(scores_ptp)  # type: ignore
    expected_metrics['logits/range_std'] = np.std(scores_ptp)  # type: ignore
    expected_metrics['logits/diff_1_to_2_mean'] = np.mean(diff_places[:, 0])  # type: ignore
    expected_metrics['logits/diff_1_to_2_std'] = np.std(diff_places[:, 0])  # type: ignore
    expected_metrics['logits/diff_2_to_3_mean'] = np.mean(diff_places[:, 1])  # type: ignore
    expected_metrics['logits/diff_2_to_3_std'] = np.std(diff_places[:, 1])  # type: ignore
    expected_metrics['logits/choice_1_rank_mean'] = np.mean(ranks[:, 0])  # type: ignore
    expected_metrics['logits/choice_2_rank_mean'] = np.mean(ranks[:, 1])  # type: ignore
    expected_metrics['logits/choice_3_rank_mean'] = np.mean(ranks[:, 2])  # type: ignore
    expected_metrics['logits/choice_1_rank_std'] = np.std(ranks[:, 0])  # type: ignore
    expected_metrics['logits/choice_2_rank_std'] = np.std(ranks[:, 1])  # type: ignore
    expected_metrics['logits/choice_3_rank_std'] = np.std(ranks[:, 2])  # type: ignore

    expected_predictions = [
        evaluation.serialize_prediction(
            idx=0,
            prediction=choices[preds[0]],
            target=ds[0]['output'],
            input_seq=ds[0]['prompt'],
            choice_logits={i: pred_scores[0][i] for i, choice in enumerate(choices)}
        ),
        evaluation.serialize_prediction(
            idx=1,
            prediction=choices[preds[1]],
            target=ds[1]['output'],
            input_seq=ds[1]['prompt'],
            choice_logits={i: pred_scores[1][i] for i, choice in enumerate(choices)}
        ),
    ]
    expected_predictions = list(map(json.loads, expected_predictions))

    metrics_path, preds_path = evaluation.evaluate(
        predictions=predictions,
        choices=choices,
        source_dataset=ds,
        tokenized_dataset=tokenized,
        out_path=Path(tmpdir)
    )

    assert metrics_path.exists()
    assert preds_path.exists()

    actual_metrics = json.loads(metrics_path.read_text('utf-8'))
    assert set(actual_metrics) == set(expected_metrics)
    for k, expected in expected_metrics.items():
        assert math.isclose(actual_metrics[k], expected), f"{k}: {actual_metrics[k]}!= {expected}"

    predictions = list(map(json.loads, preds_path.read_text('utf-8').splitlines(False)))

    for actual, expected in zip(predictions, expected_predictions):
        assert actual == expected

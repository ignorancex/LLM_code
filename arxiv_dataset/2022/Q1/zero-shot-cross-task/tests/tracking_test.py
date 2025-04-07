import json

import pandas as pd
import pytest
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, T5Config
from datasets import load_dataset
from pathlib import Path
from omegaconf import OmegaConf
from src.common.util import PROJECT_ROOT
import shutil
from promptsource.templates import DatasetTemplates
from functools import partial
import yaml
from src import tracking
from src.common import sanitize_name


def test_get_metrics_for_wandb(tmpdir):
    pred_path = Path(tmpdir).joinpath("preds.jsonl")
    met_path = Path(tmpdir).joinpath("metrics.json")
    with met_path.open("w", encoding='utf-8') as f:
        json.dump({"accuracy": 100}, f)
    choices = ["Yes", "Maybe", "No"]
    choice_logits = [
        {
            "choice_logits": {"0": -32.75, "1": -89.0, "2": -55.22}
        },
        {
            "choice_logits": {"0": -32.92, "1": -88.09, "2": -55.96}
        }
    ]
    data = [{
        "id"   : 0, "prediction": "Yes", "target": "Yes",
        "input": "Suppose The Parma trolleybus system (Italian: \"Rete filoviaria di Parma\" ) forms part of the public transport network of the city and \"comune\" of Parma, in the region of Emilia-Romagna, northern Italy. In operation since 1953, the system presently comprises four urban routes. Can we infer that \"The trolleybus system has over 2 urban routes\"? Yes, no, or maybe?"
    }, {
        "id"   : 1, "prediction": "Yes", "target": "Maybe",
        "input": "Suppose Alexandra Lendon Bastedo (9 March 1946 \u2013 12 January 2014) was a British actress, best known for her role as secret agent Sharron Macready in the 1968 British espionage/science fiction adventure series \"The Champions\". She has been cited as a sex symbol of the 1960s and 1970s. Bastedo was a vegetarian and animal welfare advocate. Can we infer that \"Sharron Macready was a popular character through the 1980\'s.\"? Yes, no, or maybe?"

    }]

    expected_data = []
    with pred_path.open('w', encoding='utf-8') as f:
        for line, choice_info in zip(data, choice_logits):
            f.write(json.dumps({**line, **choice_info}) + "\n")
            line_record = line
            for choice, (choice_id, logit) in zip(choices,
                                                  choice_info['choice_logits'].items()):
                line_record[f"choice_{choice_id}"] = choice
                line_record[f"choice_{choice_id}_logit"] = logit
            line_record['correct'] = line_record['prediction'] == line_record['target']
            expected_data.append(line_record)

    metrics, result = tracking.get_metrics_for_wandb(met_path, pred_path, choices=choices)

    assert set(metrics) == {
        "accuracy"
    }
    assert metrics['accuracy'] == 100

    expected = pd.DataFrame.from_records(expected_data)
    result = result.reindex(sorted(result.columns), axis=1)
    expected = expected.reindex(sorted(expected.columns), axis=1).sort_values(by=['id'])
    pd._testing.assert_frame_equal(result, expected)

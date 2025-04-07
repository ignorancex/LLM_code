import pytest
from pathlib import Path
from src import experiment
from omegaconf import OmegaConf
from transformers import AutoTokenizer, T5ForConditionalGeneration
from datasets import set_caching_enabled
from unittest.mock import patch
from functools import partial
from promptsource.templates import DatasetTemplates


@pytest.mark.parametrize("force", [True, False], ids=["Force", "No Force"])
@pytest.mark.parametrize("existing_dir", ["A", "Prompt"], ids=["No Conflict", "Conflict"])
@pytest.mark.parametrize("subset", [None, "cb"], ids=["No Subset", "Subset"])
def test_single_experiment(tmpdir, force, existing_dir, subset):
    set_caching_enabled(False)
    tmpdir_path = Path(tmpdir)

    cfg = OmegaConf.create({
        "task"       : "Test",
        "model_name" : "Test",
        "force"      : force,
        "batch_size" : 1,
        "beams"      : 4,
        "evaluation" : {
            "force_generation"    : False,
            "length_normalization": False,
        },
        "num_proc"   : 1,
        "cuda_device": 0

    })

    dataset_name = "anli" if not subset else "super_glue"
    split = "train[:10]" if subset else "train_r1[:10]"

    prompt = DatasetTemplates('anli')["does it follow that"]
    tokenizer = AutoTokenizer.from_pretrained('t5-small')

    tmpdir_path.joinpath(existing_dir).mkdir(parents=True)

    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(0)
    model.eval()

    if not force and existing_dir == "Prompt":
        with pytest.raises(ValueError):
            experiment.single_experiment(
                cfg,
                prompt=prompt,
                dataset_name=dataset_name,
                split=split,
                model=model,
                tokenizer=tokenizer,
                experiment_name="Test",
                prompt_file_name="Prompt",
                seeds={},
                subset=subset,
                working_dir=tmpdir_path
            )
        return

    ds, results_path = experiment.single_experiment(
        cfg,
        prompt=prompt,
        dataset_name=dataset_name,
        split=split,
        model=model,
        tokenizer=tokenizer,
        experiment_name="Test",
        prompt_file_name="Prompt",
        seeds={},
        subset=subset,
        working_dir=tmpdir_path
    )
    assert results_path.exists()
    pred_path = results_path.joinpath('predictions.jsonl')
    metrics_path = results_path.joinpath('metrics.json')
    assert pred_path.exists()
    assert metrics_path.exists()
    set_caching_enabled(True)
from metricx23 import models
import transformers
import datasets
import pandas as pd

from .doc_scorer import DocScorer


class MetricXScorer(DocScorer):
    def __init__(self, model_name="google/metricx-23-xl-v2p0", batch_size=1):
        self.maximum_val = 0.0
        self.minimum_val = -25.0
        self.batch_size = batch_size
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl")
        self.model = models.MT5ForRegression.from_pretrained(model_name)
        self.model.cuda()
        self.model.eval()

    def _make_input(self, example):
        example["input"] = (
            "candidate: "
            + example["hypothesis"]
            + " reference: "
            + example["reference"]
        )
        return example

    def _tokenize(self, example):
        return self.tokenizer(
            example["input"], max_length=1024, truncation=True, padding=False
        )

    def _remove_eos(self, example):
        example["input_ids"] = example["input_ids"][:-1]
        example["attention_mask"] = example["attention_mask"][:-1]
        return example

    def _process_dataset(self, sources, translations, references):
        ds = datasets.Dataset.from_pandas(
            pd.DataFrame(
                data=[
                    {"hypothesis": y, "reference": x}
                    for x, y in zip(references, translations)
                ]
            )
        )
        ds = ds.map(self._make_input)
        ds = ds.map(self._tokenize)
        ds = ds.map(self._remove_eos)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=0,
            output_all_columns=True,
        )
        return ds

    def get_scores(
        self,
        translations: list[str],
        sources: list[str] | None = None,
        references: list[str] | None = None,
    ):
        assert references is not None, "References are required for CHRF"
        ds = self._process_dataset(sources, translations, references)

        training_args = transformers.TrainingArguments(
            per_device_eval_batch_size=self.batch_size,
            output_dir="./",
            dataloader_pin_memory=False,
        )
        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
        )

        predictions, _, _ = trainer.predict(test_dataset=ds)  # type: ignore
        return predictions


class MetricXQEScorer(MetricXScorer):

    def __init__(self, model_name="google/metricx-23-qe-xl-v2p0", batch_size=1):
        self.maximum_val = 0.0
        self.minimum_val = -25.0
        self.batch_size = batch_size
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl")
        self.model = models.MT5ForRegression.from_pretrained(model_name)
        self.model.cuda()
        self.model.eval()

    def _make_input(self, example):
        example["input"] = (
            "candidate: "
            + example["hypothesis"]
            + " reference: "
            + example["reference"]
        )
        return example

    def _process_dataset(self, sources, translations, references):
        ds = datasets.Dataset.from_pandas(
            pd.DataFrame(
                data=[
                    {"hypothesis": y, "source": x}
                    for x, y in zip(sources, translations)
                ]
            )
        )
        ds = ds.map(self._make_input)
        ds = ds.map(self._tokenize)
        ds = ds.map(self._remove_eos)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=0,
            output_all_columns=True,
        )
        return ds

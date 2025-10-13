import torch
from tqdm import tqdm
from comet import download_model, load_from_checkpoint

from .doc_scorer import DocScorer


def build_embeddings(sources, translations, comet_model, batch_size):
    src_batches = [
        sources[i : i + batch_size] for i in range(0, len(sources), batch_size)
    ]
    src_inputs = [comet_model.encoder.prepare_sample(batch) for batch in src_batches]
    mt_batches = [
        translations[i : i + batch_size]
        for i in range(0, len(translations), batch_size)
    ]
    mt_inputs = [comet_model.encoder.prepare_sample(batch) for batch in mt_batches]

    src_embeddings = []
    with torch.no_grad():
        for batch in src_inputs:
            input_ids = batch["input_ids"].to(comet_model.device)
            attention_mask = batch["attention_mask"].to(comet_model.device)
            src_embeddings.append(
                comet_model.get_sentence_embedding(input_ids, attention_mask)
            )
    src_embeddings = torch.vstack(src_embeddings)

    mt_embeddings = []
    with torch.no_grad():
        for batch in tqdm(mt_inputs, desc="Encoding sentences...", dynamic_ncols=True):
            input_ids = batch["input_ids"].to(comet_model.device)
            attention_mask = batch["attention_mask"].to(comet_model.device)
            mt_embeddings.append(
                comet_model.get_sentence_embedding(input_ids, attention_mask)
            )
    mt_embeddings = torch.vstack(mt_embeddings)

    return src_embeddings, mt_embeddings


class CometScorer(DocScorer):
    def __init__(
        self,
        model_name="Unbabel/wmt22-comet-da",
        batch_size=8,
        device_id=0,
    ):
        checkpoint_path = download_model(model_name)
        self.model = load_from_checkpoint(checkpoint_path)
        self.batch_size = batch_size
        self.device_id = device_id

    def get_scores(
        self,
        translations: list[str],
        sources: list[str] | None = None,
        references: list[str] | None = None,
    ):
        assert sources is not None, "Must provide sources"
        assert references is not None, "Must provide references"

        seg_scores = self.model.predict(
            [
                {"mt": y, "ref": z, "src": x}
                for x, y, z in zip(sources, translations, references)
            ],
            batch_size=self.batch_size,
            gpus=1,
            progress_bar=True,
            devices=[self.device_id],
        )["scores"]
        return seg_scores

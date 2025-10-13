from comet import download_model, load_from_checkpoint

from .doc_scorer import DocScorer


class CometQEScorer(DocScorer):
    def __init__(
        self,
        model_name="Unbabel/wmt23-cometkiwi-da-xl",
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
        seg_scores = self.model.predict(
            [{"mt": y, "src": x} for x, y in zip(sources, translations)],
            batch_size=self.batch_size,
            gpus=1,
            progress_bar=True,
            devices=[self.device_id],
        )["scores"]
        return seg_scores

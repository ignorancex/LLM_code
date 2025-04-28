from typing import List, Tuple, Union

import torch
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

from open_biomed.data import Text
from open_biomed.models.functional_model import TextEncoder
from open_biomed.utils.config import Config
from open_biomed.utils.collator import Collator
from open_biomed.utils.featurizer import Featurized, TextFeaturizer, TextTransformersFeaturizer

class PretrainedLMForTextEncoding(TextEncoder):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)
        self.model = AutoModel.from_pretrained(model_cfg.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name_or_path)
        self.featurizer = TextTransformersFeaturizer(self.tokenizer, model_cfg.max_length)
        self.collator = DataCollatorWithPadding(self.tokenizer, padding=True)

    def get_text_processor(self) -> Tuple[TextFeaturizer, Collator]:
        return self.featurizer, self.collator

    def encode_text(self, text: Union[List[Text], Featurized[Text]]) -> torch.Tensor:
        if isinstance(text, list):
            text = self.collator([self.featurizer(t) for t in text]).to(self.model.device)
        return self.model(**text).last_hidden_state
        
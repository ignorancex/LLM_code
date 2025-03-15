import torch
from torch import nn
from transformers.models.roberta.modeling_roberta import (
    RobertaForSequenceClassification,
)


class RobertaClassificationHeadLinear(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceClassificationLinear(RobertaForSequenceClassification):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.classifier = RobertaClassificationHeadLinear(config)

        self.init_weights()

import torch
import torch.nn as nn
from transformers import BertTokenizer, get_linear_schedule_with_warmup, BertConfig, BertForMaskedLM


class SimpleModel(nn.Module):
    def __init__(self, model_path, ent_num, args):
        super().__init__()
        self.args = args
        self.bert_lm = BertForMaskedLM.from_pretrained(model_path)
        config = self.bert_lm.config
        self.ent_vocab = nn.Embedding(ent_num, config.hidden_size)

        self._cls_bn = nn.BatchNorm1d(num_features=1)

    def cls_bn(self, x):
        return self._cls_bn(x.unsqueeze(1)).squeeze(1)

    def get_ent_logits(self, last_hidden_state, ent_emb=None):
        if ent_emb is None:
            ent_emb = self.ent_vocab.weight

        hidden_state = self.bert_lm.cls.predictions.transform(last_hidden_state)
        return self.cls_bn(hidden_state.matmul(ent_emb.T))
import torch
import torch.nn as nn
from transformers import BertModel

from h02_bert_embeddings.bert_per_word import BertPerWordModel
from utils import constants


class BertPerWordForMaskedLM(nn.Module):
    def __init__(self, bert_option, vocab_map, mask_idx):
        super().__init__()
        self.bert_option = bert_option
        self.mask_idx = mask_idx
        self.vocab_map = vocab_map
        self.vocab_size = len(vocab_map)
        self.hidden_size_bert = 768
        self.hidden_size_inner = 200

        self.bert = self.get_bert(bert_option)

        self.linear_in = nn.Linear(self.hidden_size_bert, self.hidden_size_inner)
        self.linear_out = nn.Linear(self.hidden_size_inner, self.vocab_size)
        self.relu = nn.ReLU()

    @staticmethod
    def get_bert(bert_option):
        model = BertModel.from_pretrained(bert_option)
        return model

    def forward(self, x, attention_mask, mappings):
        batch_size = x.size(0)
        longest_token_sent = mappings.size(1)

        hidden_states_per_word = torch.zeros(
            (batch_size, longest_token_sent, self.hidden_size_bert)).to(device=constants.device)
        mask_start = torch.zeros(batch_size).long().to(device=constants.device)

        with torch.no_grad():
            for mask_pos in range(0, longest_token_sent):
                mask_sizes = mappings[:, mask_pos]
                use_rows = (mask_sizes != -1) & (mask_sizes != 0)
                if not use_rows.any():
                    break

                word_idxs = self.get_word_idxs(mask_start, mask_sizes, use_rows)

                hidden_states_bpe = self.get_masked_hidden_states(x, attention_mask, use_rows, word_idxs)
                hidden_state_size = hidden_states_bpe.size(-1)

                hidden_states_filtered = self.filter_word_values(hidden_states_bpe, word_idxs)
                hidden_states_per_word[use_rows, mask_pos] = \
                    hidden_states_filtered.sum(dim=1) / \
                    mask_sizes[use_rows].unsqueeze(-1).repeat(1, hidden_state_size).float()

                mask_start += mask_sizes

        hidden = self.relu(self.linear_in(hidden_states_per_word.detach()))
        logits = self.linear_out(hidden)

        return logits

    def get_word_idxs(self, mask_start, mask_sizes, use_rows):
        mask_start_pos = mask_start[use_rows].clone()

        mask_idxs = []
        for i, (sent_start, sent_size) in enumerate(zip(mask_start_pos, mask_sizes[use_rows])):
            mask_idxs += [(i, sent_start.item() + x) for x in range(sent_size)]
        return list(zip(*mask_idxs))

    def get_masked_hidden_states(self, x, attention_mask, use_rows, word_idxs):
        x_pos = x[use_rows].clone()
        attention_mask = attention_mask[use_rows].clone()

        x_pos[word_idxs] = self.mask_idx
        # attention_mask_pos[word_idxs] = 0

        output, _ = self.bert(x_pos, attention_mask=attention_mask)
        hidden_states_bpe = output[:, 1:-1]

        return hidden_states_bpe

    def filter_word_values(self, hidden_states_bpe, word_idxs):
        hidden_states_filtered = \
            torch.zeros_like(hidden_states_bpe).float().to(device=constants.device)
        hidden_states_filtered[word_idxs] = hidden_states_bpe[word_idxs]

        return hidden_states_filtered

    def get_target_tensor(self, batch):
        batch_size = len(batch)
        longest_token_sent = max([len(x) for x in batch])

        target_ids = torch.ones((batch_size, longest_token_sent)).long() * self.vocab_map['<PAD>']
        for i, sentence in enumerate(batch):
            sentence_len = len(sentence)
            target_ids[i, :sentence_len] = torch.tensor(self.get_ids_from_tokens(sentence))

        return target_ids.to(device=constants.device)

    def get_ids_from_tokens(self, tokens):
        return [self.vocab_map[x] if x in self.vocab_map else self.vocab_map['<UNK>'] for x in tokens]

    def save(self, fname):
        torch.save({
            'kwargs': self.get_args(),
            'model_state_dict': self.state_dict(),
        }, fname)

    def get_args(self):
        return {
            'bert_option': self.bert_option,
            'vocab_map': self.vocab_map,
            'mask_idx': self.mask_idx,
        }

    @classmethod
    def load(cls, fname, device):
        checkpoints = cls.load_checkpoint(fname, device)
        model = cls(**checkpoints['kwargs'])
        model.load_state_dict(checkpoints['model_state_dict'])
        return model

    @classmethod
    def load_checkpoint(cls, fname, device):
        return torch.load(fname, map_location=device)

    @classmethod
    def load_vocab(cls, fname):
        checkpoints = cls.load_checkpoint(fname, 'cpu')
        return checkpoints['kwargs']['vocab_map']

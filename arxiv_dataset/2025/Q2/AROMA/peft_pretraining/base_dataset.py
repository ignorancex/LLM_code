from dataclasses import dataclass
import torch
from transformers import PreTrainedTokenizer
from typing import Dict, Sequence

IGNORE_INDEX = -100

@dataclass
class DataCollatorForSupervisedDataset:
    
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] 
                                for key in ("input_ids", "labels"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=IGNORE_INDEX
        )
        
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

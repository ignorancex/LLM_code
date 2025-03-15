
import torch
from torch.utils.data import Dataset
class Data(Dataset):
    def __init__(self, df, tokenizer, config, max_len=256):
        self.df = df
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.config = config
    
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        row = self.df.iloc[index]
        claim, context, label, ids = self.get_input_data(row)
        
        encoding = self.tokenizer.encode_plus(
            claim,
            context,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt', 
        )
        
        return {
            'id': ids,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_masks': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long),
        }


    def labelencoder(self, text):
        label_map = {
            3: {'NEI': 0, 'SUPPORTED': 1, 'REFUTED': 2},
            2: {'SUPPORTED': 0, 'REFUTED': 1} 
        }
        return label_map[self.config.n_classes].get(text, 1)

    def get_input_data(self, row):
        claim = row['claim']
        context = row['context']
        ids = row['id']
        label = self.labelencoder(row['verdict'])
        
        return str(claim), str(context), label, ids
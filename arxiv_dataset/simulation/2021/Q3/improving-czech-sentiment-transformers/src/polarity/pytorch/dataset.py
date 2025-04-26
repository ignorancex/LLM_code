import torch

from torch.utils.data import Dataset, DataLoader

class PolarityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        if len(texts) != len(labels):
            raise Exception(f"Lens of texts: {len(texts)} does not correspond to lends of labels: {len(labels)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        ids_tensor = self.tokenizer.encode_plus(text,
                                                max_length=self.max_len,
                                                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                                return_token_type_ids=False,
                                                pad_to_max_length=True,
                                                return_attention_mask=True,
                                                truncation=True,
                                                return_tensors='pt',  # Return PyTorch tensors
                                                )

        return {
            'text': text,
            'input_ids': ids_tensor['input_ids'].flatten(),
            'attention_mask': ids_tensor['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def build_data_loader(df, tokenizer, max_len, batch_size, shuffle=True):
    dataset = PolarityDataset(df.text.to_numpy(),
                              df.label.to_numpy(),
                              tokenizer,
                              max_len)
    # set num_workers to 0 while in debug, otherwise set to 6
    return DataLoader(dataset, batch_size, num_workers=4)
    # return DataLoader(dataset, batch_size, num_workers=0)

from torch.utils.data import Dataset
import csv


class ChatData(Dataset):
    def __init__(self, path: str, tokenizer):


        self.X = []
        with open('True.csv', encoding='UTF-8') as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            for row in f_csv:
                site = row[1].find('Trump')
                if site == -1:
                    continue
                self.X.append(row[1][0:site + 5])
                self.X.append(row[1][site + 5:-1])

        for idx, i in enumerate(self.X):
            try:
                self.X[idx] = "<startofstring> " + i + " <bot>: " + self.X[idx + 1] + " <endofstring>"
            except:
                break

        self.X = self.X[:5000]

        print(self.X[0])

        self.X_encoded = tokenizer(self.X, max_length=40, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])
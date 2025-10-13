import audobject
import torch

class LabelEncoder(audobject.Object):
    def __init__(self, labels):
        self.labels = sorted(labels)
        codes = range(len(self.labels))
        self.inverse_map = {code: label for code,
                            label in zip(codes, self.labels)}
        self.map = {label: code for code,
                    label in zip(codes, self.labels)}

    def encode(self, x):
        return self.map[x]

    def decode(self, x):
        return self.inverse_map[x]
    
    def __call__(self, x):
        return self.encode(x)

    def get_predictions(self, x):
        if torch.is_tensor(x):
            x = x.numpy()
        return x.argmax(-1).squeeze()

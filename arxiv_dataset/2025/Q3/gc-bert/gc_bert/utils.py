import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def to_torch_sparse(x):
    coo = x.tocoo()
    row = torch.from_numpy(coo.row.astype(np.float32)).to(torch.long)
    col = torch.from_numpy(coo.col.astype(np.float32)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)

    val = torch.from_numpy(coo.data.astype(np.float64)).to(torch.float32)

    return torch.sparse.DoubleTensor(edge_index, val, torch.Size(coo.shape))

def split(n, seed=1):
    train_idx, test_idx = train_test_split(np.arange(n), train_size=0.7, random_state=seed)
    valid_idx, test_idx = train_test_split(test_idx, train_size=1/3, random_state=seed)

    idx_train = np.zeros(n).astype(bool)
    idx_train[train_idx] = True

    idx_valid = np.zeros(n).astype(bool)
    idx_valid[valid_idx] = True

    idx_test = np.zeros(n).astype(bool)
    idx_test[test_idx] = True
    return idx_train, idx_valid, idx_test
    

def vectorize_texts(texts, dim=512, min_df=0.001, max_df=0.5, to_sparse=True):
    vectorizer = TfidfVectorizer(
        strip_accents='ascii', max_df=max_df, min_df=min_df, max_features=dim
    )
    X = vectorizer.fit_transform(
        texts
    )
    if to_sparse:
        X = to_torch_sparse(X)
    else:
        X = torch.tensor(X.todense(), dtype=torch.float32)
    return X


def accuracy(preds, labels):
    if len(preds.shape) == 2:
        preds = preds.max(1)[1].type_as(labels)
    correct = preds.eq(labels).to(torch.float32)
    correct = correct.sum()
    return correct / len(labels)

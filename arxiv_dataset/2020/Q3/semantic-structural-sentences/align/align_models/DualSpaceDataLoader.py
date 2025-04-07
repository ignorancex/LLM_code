# TODO: refactor such that a giant set of embeddings is read in, and the data loader does the train test val splits
import numpy as np
import torch


def cov(m, rowvar=True, inplace=False):
    """
    Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
        inplace: Create copies of the tensors or not.

    Returns:
        The covariance matrix of the variables.
    """
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


class DualSpaceDataLoader(object):

    def __init__(self,
                 train_src_emb_path, train_trg_emb_path,
                 valid_src_emb_path, valid_trg_emb_path,
                 train_batch_sz, _normalize_vecs=None):
        self.train_src_emb = torch.load(train_src_emb_path)
        self.train_trg_emb = torch.load(train_trg_emb_path)
        self.valid_src_emb = torch.load(valid_src_emb_path)
        self.valid_trg_emb = torch.load(valid_trg_emb_path)
        self.train_batch_sz = train_batch_sz
        self.normalize_vecs = _normalize_vecs
        self.normalize_embeddings()

    def get_norm_method(self):
        return self.normalize_vecs

    def normalize_embeddings(self):
        if self.normalize_vecs == 'whiten':
            print("Normalizing embeddings with {}".format(self.normalize_vecs))
            self._center_embeddings()
            self._whiten_embeddings()
        elif self.normalize_vecs == 'mean':
            print("Normalizing embeddings with {}".format(self.normalize_vecs))
            self._center_embeddings()
        elif self.normalize_vecs == 'both':
            print("Normalizing embeddings with {}".format(self.normalize_vecs))
            self._scale_embeddings()
            self._center_embeddings()
        else:
            print('Warning: no normalization performed')

    def _center_embeddings(self):
        train_src_mean = self.train_src_emb.mean(axis=0)
        self.train_src_emb = self.train_src_emb - train_src_mean
        self.valid_src_emb = self.valid_src_emb - train_src_mean
        train_trg_mean = self.train_trg_emb.mean(axis=0)
        self.train_trg_emb = self.train_trg_emb - train_trg_mean
        self.valid_trg_emb = self.valid_trg_emb - train_trg_mean
        self.centered = True

    def _scale_embeddings(self):
        train_src_norm = self.train_src_emb.norm(p=2, dim=1, keepdim=True)
        self.train_src_emb = self.train_src_emb.div(train_src_norm)
        valid_src_norm = self.valid_src_emb.norm(p=2, dim=1, keepdim=True)
        self.valid_src_emb = self.valid_src_emb.div(valid_src_norm)
        train_trg_norm = self.train_trg_emb.norm(p=2, dim=1, keepdim=True)
        self.train_trg_emb = self.train_trg_emb.div(train_trg_norm)
        valid_trg_norm = self.valid_trg_emb.norm(p=2, dim=1, keepdim=True)
        self.valid_trg_emb = self.valid_trg_emb.div(valid_trg_norm)

    def _whiten_embeddings(self):
        if not self.centered:
            raise ValueError("Whitening needs centering to be done in advance")
        src_cov = cov(self.train_src_emb.T)
        u_src, s_src, v_src = torch.svd(src_cov)
        w_src = (v_src.T / torch.sqrt(s_src)).T
        self.train_src_emb = self.train_src_emb @ w_src.T
        self.valid_src_emb = self.valid_src_emb @ w_src.T
        trg_cov = cov(self.train_trg_emb.T)
        u_trg, s_trg, v_trg = torch.svd(trg_cov)
        w_trg = (v_trg.T / torch.sqrt(s_trg)).T
        self.train_trg_emb = self.train_trg_emb @ w_trg.T
        self.valid_trg_emb = self.valid_trg_emb @ w_trg.T

    def training_minibatch(self):
        if self.train_batch_sz > 0:
            idx = np.random.randint(self.train_src_emb.shape[0], size=(self.train_batch_sz,))
        else:
            idx = list(range(self.train_src_emb.shape[0]))
        train_src_emb = self.train_src_emb[idx]
        train_tgt_emb = self.train_trg_emb[idx]
        return torch.tensor(idx), train_src_emb, train_tgt_emb

    def get_training_batches(self):
        if self.train_batch_sz == 0:
            return self.training_minibatch()
        else:
            batch_list = []
            for j in range(self.train_src_emb.shape[0] // self.train_batch_sz):
                batch_list.append(self.training_minibatch())
            return batch_list

    def get_all_validation(self):
        return self.valid_src_emb, self.valid_trg_emb

    def get_all_training(self):
        return self.train_src_emb, self.train_trg_emb

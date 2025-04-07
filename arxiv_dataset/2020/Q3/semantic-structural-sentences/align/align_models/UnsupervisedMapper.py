import torch
import torch.nn as nn


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


def topk_mean(_m, k):
    n = _m.shape[0]
    ans = torch.zeros(n, dtype=_m.dtype)
    if k <= 0:
        return ans
    ind_0 = torch.arange(n)
    ind_1 = torch.empty(n, dtype=torch.int16)
    minimum = _m.min().item()
    for j in range(k):
        cur_max = _m.argmax(dim=1)
        ind_1.insert(cur_max)  # TODO: this is going to need some help
        ans += _m[ind_0, ind_1]
        _m[ind_0, ind_1] = minimum
    return ans / k


class UnsupervisedMapper(nn.Module):
    def __init__(self, emb_src, emb_trg, metric='euclidean', normalize_vecs="both",
                 mapping='unsupervised', orthogonal=True, unconstrained=False, unsupervised_vocab=0,
                 csls_neighborhood=5, direction="forward",
                 stochastic_initial=0.1, stochastic_multiplier=2.0, stochastic_interval=50, self_learning=True):
        super(UnsupervisedMapper, self).__init__()
        self.emb_src = emb_src
        self.emb_trg = emb_trg
        self.metric = metric
        self.normalize_vecs = normalize_vecs
        self.centered = False
        self.mapping = mapping
        self.orthogonal = orthogonal
        self.unconstrained = unconstrained
        self.unsupervised_vocab = unsupervised_vocab
        self.csls_neighborhood = csls_neighborhood
        self.direction = direction
        self.src_ind = None  # Initialized source anchors (chosen in build_unsupervised_anchors)
        self.trg_ind = None  # Initialized target anchors (chosen in build_unsupervised_anchors)

        # Stochastic parameters control the 'noise' in training, allowing the process to jump out of local minima
        self.stochastic_initial = stochastic_initial
        self.stochastic_multiplier = stochastic_multiplier
        self.stochastic_interval = stochastic_interval
        self.self_learning = self_learning

    def normalize_embeddings(self):
        if self.normalize_vecs:
            print("Normalizing embeddings with {}".format(self.normalize_vecs))
        if self.normalize_vecs == 'whiten':
            self._center_embeddings()
            self._whiten_embeddings()
        elif self.normalize_vecs == 'mean':
            self._center_embeddings()
        elif self.normalize_vecs == 'both':
            self._center_embeddings()
            self._scale_embeddings()
        else:
            print('Warning: no normalization performed')

    def _center_embeddings(self):
        self.emb_src -= self.emb_src.mean(axis=0)
        self.emb_trg -= self.emb_trg.mean(axis=0)
        self.centered = True

    @staticmethod
    def center_vectors(self, a, b):
        a -= a.mean(axis=0)
        b -= b.mean(axis=0)
        return a, b

    def _scale_embeddings(self):
        self.emb_src = self.emb_src / torch.norm(self.emb_src, p=2, dim=1)
        self.emb_trg = self.emb_trg / torch.norm(self.emb_trg, p=2, dim=1)

    @staticmethod
    def scale_vectors(self, a, b):
        a = a / torch.norm(a, p=2, dim=1)
        b = b / torch.norm(b, p=2, dim=1)
        return a, b

    def _whiten_embeddings(self):
        if not self.centered:
            raise ValueError("Whitening needs centering to be done in advance")
        cov_s = cov(self.emb_src.T)
        u_src, s_src, v_src = torch.svd(cov_s)
        w_src = (v_src.T / torch.sqrt(s_src)).T
        self.emb_src = self.emb_src @ w_src.T
        cov_t = cov(self.emb_trg.T)
        u_trg, s_trg, v_trg = torch.svd(cov_t)
        w_trg = (v_trg.T / torch.sqrt(s_trg)).T
        self.emb_trg = self.emb_trg @ w_trg.T

    def build_unsupervised_anchors(self):
        if self.mapping == 'unsupervised':
            sim_size = min(self.emb_src.size[0], self.emb_trg.size[0]) if self.unsupervised_vocab <= 0 else min(
                self.emb_src.size[0], self.emb_trg.size[0], self.unsupervised_vocab
            )
            u, s, vt = torch.svd(self.emb_src[:sim_size])
            src_sim = (u * s).dot(u.T)
            u, s, vt = torch.svd(self.emb_trg[:sim_size])
            trg_sim = (u * s).dot(u.T)
            src_sim.sort(dim=1)
            trg_sim.sort(dim=1)
            src_sim, trg_sim = self.center_vectors(src_sim, trg_sim)
            src_sim, trg_sim = self.scale_vectors(src_sim, trg_sim)
            sim = src_sim.dot(trg_sim.T)
            if self.csls_neighborhood > 0:
                knn_sim_fwd = topk_mean(sim, k=self.csls_neighborhood)
                knn_sim_bwd = topk_mean(sim.T, k=self.csls_neighborhood)
                sim -= knn_sim_fwd / 2 + knn_sim_bwd / 2  # TODO: going to need dim checks here
            if self.direction == "forward":
                self.src_ind = torch.arange(sim_size)
                self.trg_ind = sim.argmax(dim=1)
            elif self.direction == "backward":
                self.src_ind = sim.argmax(dim=0)
                self.trg_ind = torch.arange(sim_size)
            elif self.direction == "union":
                self.src_ind = torch.cat([torch.arange(sim_size), sim.argmax(dim=0)])
                self.trg_ind = torch.cat([sim.argmax(dim=1), torch.arange(sim_size)])

    def train(self):
        src_w = torch.empty_like(self.emb_src)
        trg_w = torch.empty_like(self.emb_trg)
        it = 1
        last_improvement = 0
        keep_prob = self.stochastic_initial
        end = not self.self_learning
        while True:
            if it - last_improvement > self.stochastic_interval:
                if keep_prob >= 1.0:
                    end = True
                keep_prob = min(1.0, self.stochastic_multiplier * keep_prob)
                last_improvement = it
            if self.orthogonal or not end:
                u, s, vt = torch.svd(self.emb_trg[self.trg_ind].T.dot(self.emb_src[self.src_ind]))
                w = torch.dot(vt.T, u.T)
                torch.dot(self.emb_src, w, out=src_w)
                trg_w[:] = self.emb_trg
            elif self.unconstrained:
                # TODO: finish this line of work


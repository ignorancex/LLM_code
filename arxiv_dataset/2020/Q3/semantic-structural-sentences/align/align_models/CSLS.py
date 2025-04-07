# TODO: Cross-domain Similarity Local Scaling for NN evaluations to mitigate hubbiness
import torch
import torch.nn as nn
import faiss


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, embedding_dim


def get_faiss_nearest_neighbours(emb_src, emb_wrt, k, use_gpu=True, gpu_device=0):
    """
    Gets source points'/embeddings' nearest neighbours with respect to a set of target embeddings.
    inputs:
        :param emb_src (np.ndarray) : the source embedding matrix
        :param emb_wrt (np.ndarray) : the embedding matrix in which nearest neighbours are to be found
        :param k (int) : the number of nearest neightbours to find
        :param use_gpu (bool) : true if the gpu is to be used
        :param gpu_device (int) : the GPU to be used
    outputs:
        :returns distance (np.ndarray) : [len(emb_src), k] matrix of distance of
            each source point to each of its k nearest neighbours
        :returns indices (np.ndarray) : [len(emb_src), k] matrix of indices of
            each source point to each of its k nearest neighbours
    """
    if use_gpu:
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = gpu_device
        index = faiss.GpuIndexFlatIP(res, emb_wrt.shape[1], cfg)
    else:
        index = faiss.IndexFlatIP(emb_wrt.shape[1])
    print('Building Faiss index')
    index.add(emb_wrt.cpu().detach().numpy())
    print('... Done!')
    res = index.search(emb_src.cpu().detach().numpy(), k)
    return torch.tensor(res).cuda()


def get_mean_similarity(emb_src, emb_wrt, k, use_gpu=True, gpu_device=0):
    """
    Gets the mean similarity of source embeddings with respect to a set of target embeddings.
    inputs:
        :param emb_src (np.ndarray) : the source embedding matrix
        :param emb_wrt (np.ndarray) : the embedding matrix wrt which the similarity is to be calculated
        :param k (int) : the number of points to be used to find mean similarity
        :param use_gpu (bool) : true if the gpu is to be used
        :param gpu_device (int) : the GPU to be used
    """
    nn_dists, _ = get_faiss_nearest_neighbours(emb_src, emb_wrt, k, use_gpu, gpu_device)
    return nn_dists.mean(1)


def normalize(arr):
    """
    Normalizes a vector of vectors into a vector of unit vectors
    """
    return arr / torch.norm(arr, p=2, dim=1).unsqueeze(1)


class CSLS(nn.Module):

    def __init__(self, _model, _node_weights, _sent_weights, _k, _normalize_vecs=None):
        super().__init__()
        torch.manual_seed(17)
        self.model = _model
        self.init_nodes = _node_weights
        self.init_sents = _sent_weights
        self.node_embedding, self.node_dim = create_emb_layer(self.init_nodes, True)
        self.sent_embedding, self.sent_dim = create_emb_layer(self.init_sents, True)
        self.normalize_vecs = "norm" if _normalize_vecs else None
        self.normalize_embeddings()
        self.k = _k
        self.gpu = True
        self.gpu_device = 0
        self.source = self.node_embedding.weight.cuda()
        self.mapped_source = torch.mm(self.source, torch.transpose(self.model.mapping.weight.data, 1, 0))
        self.target = self.sent_embedding.weight.cuda()
        self.r_t = get_mean_similarity(self.mapped_source, self.target, self.k, self.gpu, self.gpu_device)
        self.r_s = get_mean_similarity(self.target, self.target, self.k, self.gpu, self.gpu_device)

    def normalize_embeddings(self):
        if self.normalize_vecs == 'norm':
            print("Normalizing embeddings with {}".format(self.normalize_vecs))
            self._scale_embeddings()
        else:
            print('Warning: no normalization performed')

    def _scale_embeddings(self):
        new = self.node_embedding.weight.detach()
        node_norm = new.norm(p=2, dim=1, keepdim=True).detach()
        node_update = new.div(node_norm)
        self.node_embedding.load_state_dict({'weight': node_update})
        sew = self.sent_embedding.weight.detach()
        sent_norm = sew.norm(p=2, dim=1, keepdim=True)
        sent_update = sew.div(sent_norm)
        self.sent_embedding.load_state_dict({'weight': sent_update})

    def get_closest_csls_matches(self):
        batch_scores = (2 * torch.mm(self.mapped_source, self.target.t())) - self.r_t - self.r_s
        return batch_scores

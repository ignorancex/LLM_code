import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

EPS=0.00001


class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss

def contrastiveloss_srcadded(z, n_way, n_support, tau):
    T1 = np.eye(n_way)
    T2 = np.ones(n_support)
    mask_pos = torch.FloatTensor(np.kron(T1, T2)).cuda()
    mask_pos = mask_pos.unsqueeze(1).repeat(1, n_support, 1).contiguous().view(n_way * n_support, -1)
    mask_neg = torch.ones([n_way * n_support, n_way * n_support]).cuda() - mask_pos
    dist_matrix = cosine_dist(z, z)
    dist_matrix = dist_matrix - dist_matrix.diag().diag()
    dist_matrix = torch.exp(dist_matrix / tau)
    pos = dist_matrix * mask_pos
    pos = pos.sum(dim=1)
    neg = dist_matrix * mask_neg
    neg = neg.sum(dim=1)
    #dist_matrix = dist_matrix.sum(dim=1)
    li = -torch.log(pos / (pos + n_way * neg))
    loss = li.sum()
    return loss



def ewn_contrastive_loss(z, mask_pos, mask_neg, mask_distract, n_s, alpha, tau):
    # equally weighted task and distractor negative contrastive loss
    bsz, featdim = z.size()
    z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    sim = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
    # sim = sim ** 3
    Sv = torch.exp(sim / tau)
    neg = (Sv * mask_neg)
    neg = alpha * (1 - mask_distract) * neg + (1 - alpha) * mask_distract * neg
    neg = 2 * neg
    neg = neg.sum(dim=1).unsqueeze(1).repeat(1, bsz)
    li = mask_pos * torch.log(Sv / (Sv + neg) + EPS)
    li = li - li.diag().diag()
    li = (1 / (n_s - 1)) * li.sum(dim=1)
    loss = -li[mask_pos.sum(dim=1) > 0].mean()
    return loss


#################  MAIN ##############
def cssf_loss(z, shots_per_way, n_way, n_ul, tau):
    # labelled positives and all negatives
    n_pos = 2
    n_l = n_way * shots_per_way
    # positive mask
    T1 = np.eye(int(n_l / n_pos))
    T2 = np.ones((n_pos, n_pos))
    mask_pos_lab = torch.FloatTensor(np.kron(T1, T2))
    T3 = torch.cat([mask_pos_lab, torch.zeros(n_l, n_ul)], dim=1)
    T4 = torch.zeros(n_ul, n_l + n_ul)
    mask_pos = torch.cat([T3, T4], dim=0).to(z.device)
    # negative mask
    T1 = 1 - np.eye(n_way)
    T2 = np.ones((shots_per_way, shots_per_way))
    mask_neg_lab = torch.FloatTensor(np.kron(T1, T2))
    T3 = torch.cat([mask_neg_lab, torch.ones(n_l, n_ul)], dim=1)
    T4 = torch.ones(n_ul, n_l + n_ul)  # dummy
    mask_neg = torch.cat([T3, T4], dim=0).to(z.device)
    T3 = torch.cat([torch.zeros(n_l, n_l), torch.ones(n_l, n_ul)], dim=1)
    mask_distract = torch.cat([T3, T4], dim=0).to(z.device)
    alpha = n_ul / (n_ul + n_l - shots_per_way)
    return ewn_contrastive_loss(z, mask_pos, mask_neg, mask_distract, n_pos, alpha, tau)

def ewn_contrastive(z, mask_pos, mask_neg, n_s, tau):
    # equally weighted task and distractor negative contrastive loss
    bsz, featdim = z.size()
    z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    sim = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
    # sim = sim ** 7
    Sv = torch.exp(sim / tau)
    neg = (Sv * mask_neg)
    neg = neg.sum(dim=1).unsqueeze(1).repeat(1, bsz)
    li = mask_pos * torch.log(Sv / (Sv + neg) + EPS)
    li = li - li.diag().diag()
    li = (1 / (n_s - 1)) * li.sum(dim=1)
    loss = -li[mask_pos.sum(dim=1) > 0].mean()
    return loss

def contrastive_loss(z, shots_per_way, n_way, tau):
    # labelled positives and all negatives
    n_pos = shots_per_way
    n_l = n_way * shots_per_way
    # positive mask
    T1 = np.eye(int(n_l / n_pos))
    T2 = np.ones((n_pos, n_pos))
    mask_pos = torch.FloatTensor(np.kron(T1, T2)).cuda()
    # negative mask
    T3 = 1 - np.eye(n_way)
    T4 = np.ones((shots_per_way, shots_per_way))
    mask_neg = torch.FloatTensor(np.kron(T3, T4)).cuda()
    return ewn_contrastive(z, mask_pos, mask_neg, n_pos, tau)

def cosine_dist(x, y):
	# x: N x D
	# y: M x D
	n = x.size(0)
	m = y.size(0)
	d = x.size(1)
	assert d == y.size(1)

	x = x.unsqueeze(1).expand(n, m, d)
	y = y.unsqueeze(0).expand(n, m, d)
	alignment = nn.functional.cosine_similarity(x, y, dim=2)
	return alignment

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def fewshot_task_loss(model, x, n_way, n_support, n_query):
    y_query = torch.from_numpy(np.repeat(range(n_way), n_query))
    y_query = y_query.cuda()
    x = x.contiguous().view(n_way * (n_support + n_query), *x.size()[2:])
    z_all_linearized = model(x)
    z_all = z_all_linearized.view(n_way, n_support + n_query, -1)
    z_support = z_all[:, :n_support]
    z_query = z_all[:, n_support:]
    z_support = z_support.contiguous()
    z_proto = z_support.view(n_way, n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
    z_query = z_query.contiguous().view(n_way * n_query, -1)

    # normalize
    z_proto = F.normalize(z_proto, dim=1)
    z_query = F.normalize(z_query, dim=1)

    scores = cosine_dist(z_query, z_proto)
    return scores, z_all_linearized

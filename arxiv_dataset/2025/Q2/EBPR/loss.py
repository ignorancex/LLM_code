import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
import ot

def sort_array(old_array, index_array):
    sorted_array = np.ones_like(old_array)
    sorted_array[index_array] = old_array
    return sorted_array

def EOT(N, OT_logits_list, epoch, lamda):
    sam_energy = torch.logsumexp(OT_logits_list / 1000, dim=1).cuda()
    Pen = nn.functional.softmax(OT_logits_list, 1).cuda()
    Pen = Pen.detach()
    N, K = Pen.shape
    Pen = Pen.T
    a = (torch.ones((K, 1)) / K).double()
    a = a.cuda()
    b = (torch.ones((N, 1)) / N).double()
    b = b.cuda()
    Pen = torch.pow(Pen, lamda).double()  # K x N
    K_ini = (1. / K)
    K_ini = (torch.tensor(K_ini)).double()
    K_ini = K_ini.cuda()
    N_ini = (sam_energy.reshape((N, 1))).double()
    N_ini = F.normalize(N_ini, p=1, dim=0)
    N_ini = N_ini.cuda()
    err = 1e3
    step = 0
    while err > 1e-1:
        a = K_ini / (Pen @ b)  # (KxN)@(N,1) = K x 1
        b_iter = N_ini / (a.T @ Pen).T  # ((1,K)@(KxN)).t() = N x 1
        if step % 5 == 0:
            err_iter = torch.abs(b / b_iter - 1)
            err_iter = torch.where(torch.isnan(err_iter), torch.full_like(err_iter, 0), err_iter)
            err = torch.sum(err_iter)
        b = b_iter
        step += 1
    Pen = (Pen * b.squeeze()).T * a.squeeze()  # N × K -> K × N
    Pen = torch.nan_to_num(Pen)

    label_ET = torch.argmax(Pen, 0)  # size N
    return Pen, label_ET

def execute_clustering(round_epoch, labeled_loader, unlabeled_loader, cluster_count, neural_net, lambda_param):
    labeled_indices, unlabeled_indices = [], []
    logits_list, labeled_labels = [], []
    
    for batch in iter(labeled_loader):
        indices = batch["index"]
        labels = batch["label"]
        data = batch["OT_data"].cuda()
        _, logits = neural_net(data, return_aux=True)
        logits = logits.detach()

        labeled_indices.append(indices)
        labeled_labels.append(labels)
        logits_list.append(logits)

    labeled_indices = torch.cat(labeled_indices).numpy().astype(int)
    labeled_labels = torch.cat(labeled_labels).numpy().astype(int)
    logits_list = torch.cat(logits_list, dim=0).cpu().tolist()

    labeled_labels = sort_array_by_index(labeled_labels, labeled_indices)
    labeled_loader.dataset.pseudo_label = labeled_labels
    
    torch.cuda.empty_cache()

    confidences, pseudo_labels, oe_logits_list, sc_labels = [], [], [], []
    for batch in iter(unlabeled_loader):
        indices = batch["index"]
        data = batch["OT_data"].cuda()
        sc_label = batch["sc_label"]
        logit, logits = neural_net(data, return_aux=True)
        logits = logits.detach()
        score = torch.softmax(logit, dim=1)
        confidence, pseudo = torch.max(score, dim=1)

        unlabeled_indices.append(indices)
        confidences.append(confidence)
        oe_logits_list.append(logits)
        sc_labels.append(sc_label)

    oe_logits_list = torch.cat(oe_logits_list, dim=0).cpu().tolist()
    logits_list.extend(oe_logits_list)
    logits_tensor = torch.tensor(logits_list).cuda()

    unlabeled_indices = torch.cat(unlabeled_indices).numpy().astype(int)
    confidences = torch.cat(confidences).cpu().numpy()
    sc_labels = torch.cat(sc_labels).numpy()

    confidences = sort_array_by_index(confidences, unlabeled_indices)
    sc_labels = sort_array_by_index(sc_labels, unlabeled_indices)
    torch.cuda.empty_cache()

    total_samples = len(logits_tensor) 

    with torch.no_grad():
        penalty_matrix, estimated_labels = EOT(total_samples, logits_tensor, round_epoch, lambda_param)

    estimated_labels = estimated_labels.tolist()
    labeled_clusters = estimated_labels[:len(labeled_loader.dataset)]
    unlabeled_clusters = estimated_labels[len(labeled_loader.dataset):]

    labeled_clusters = sort_array_by_index(labeled_clusters, labeled_indices)
    unlabeled_clusters = sort_array_by_index(unlabeled_clusters, unlabeled_indices)
    labeled_loader.dataset.cluster_id = labeled_clusters
    unlabeled_loader.dataset.cluster_id = unlabeled_clusters

    all_cluster_ids = np.concatenate([labeled_clusters, unlabeled_clusters])
    cluster_statistics = np.zeros(cluster_count)
    unique_clusters, cluster_counts = np.unique(all_cluster_ids, return_counts=True)
    for cluster_id, count in zip(unique_clusters, cluster_counts):
        cluster_statistics[cluster_id] = count

    old_labeled_pseudo_labels = labeled_loader.dataset.pseudo_label
    old_unlabeled_pseudo_labels = unlabeled_loader.dataset.pseudo_label
    all_old_pseudo_labels = np.concatenate([old_labeled_pseudo_labels, old_unlabeled_pseudo_labels]).astype(int)
    new_pseudo_labels = -1 * np.ones_like(all_old_pseudo_labels).astype(int)

    for i in range(cluster_count):
        label_in_cluster, label_counts = np.unique(all_old_pseudo_labels[all_cluster_ids == i], return_counts=True)
        cluster_size = len(all_old_pseudo_labels[all_cluster_ids == i])
        purity = label_counts / cluster_size
        if np.any(purity > 0.5):
            majority_label = label_in_cluster[purity > 0.5][0]
            new_pseudo_labels[all_cluster_ids == i] = majority_label

    unlabeled_loader.dataset.pseudo_label = new_pseudo_labels[len(labeled_loader.dataset):]

    return penalty_matrix, estimated_labels, new_pseudo_labels

def sort_array_by_index(array_to_sort, reference_indices):
    sorted_indices = np.argsort(reference_indices)
    return array_to_sort[sorted_indices]

def update__params__(sat_ema, logits_ulb_w, tau_t, p_t, label_hist):

    probs_ulb_w = torch.softmax(logits_ulb_w, dim=-1)
    max_probs_w, max_idx_w = torch.max(probs_ulb_w, dim=-1)
    tau_t = tau_t * sat_ema + (1. - sat_ema) * max_probs_w.mean()
    p_t = p_t * sat_ema + (1. - sat_ema) * probs_ulb_w.mean(dim=0)
    histogram = torch.bincount(max_idx_w, minlength=p_t.shape[0]).to(p_t.dtype)
    label_hist = label_hist * sat_ema + (1. - sat_ema) * (histogram / histogram.sum())
    return tau_t, p_t, label_hist

def SelfAdaptiveThreshold(aet_ema, logits_ulb_w, logits_ulb_s, tau_t, p_t, label_hist):

    tau_t, p_t, label_hist = update__params__(aet_ema, logits_ulb_w, tau_t, p_t, label_hist)
    
    logits_ulb_w = logits_ulb_w.detach()
    probs_ulb_w = torch.softmax(logits_ulb_w, dim=-1)
    max_probs_w, max_idx_w = torch.max(probs_ulb_w, dim=-1)
    tau_t_c = (p_t / torch.max(p_t, dim=-1)[0])
    mask = max_probs_w.ge(tau_t * tau_t_c[max_idx_w]).to(max_probs_w.dtype)

    
    return  mask, tau_t, p_t, label_hist

def compute_aug_loss(z_i, z_j, temperature=0.1):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    sim_matrix = torch.matmul(z, z.T) / temperature
    sim_matrix = F.softmax(sim_matrix, dim=1)
    
    pos_sim = torch.diag(sim_matrix, batch_size) + torch.diag(sim_matrix, -batch_size)
    loss = -torch.log(pos_sim).mean()
    return loss

def consistency_loss(logits_s, logits_w, qhat, name='ce', e_cutoff=-8, use_hard_labels=True, use_marginal_loss=True, tau=0.5):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = F.softmax(logits_w, dim=1)

        max_probs, max_idx = torch.max(pseudo_label, dim=-1)

        energy = -torch.logsumexp(logits_w, dim=1)
        mask_raw = energy.le(e_cutoff)
        mask = mask_raw.float()
        select = mask_raw.long()

        if use_marginal_loss:
            delta_logits = torch.log(qhat)
            logits_s = logits_s + tau * delta_logits

        masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask

        return masked_loss.mean(), mask.mean(), select, max_idx.long(), mask_raw

    else:
        assert Exception('Not Implemented consistency_loss')

def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss
    

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 


def bsp_loss(feature):
    train_bs = feature.size(0) // 2
    feature_s = feature.narrow(0, 0, train_bs)
    feature_t = feature.narrow(0, train_bs, train_bs)
    _, s_s, _ = torch.svd(feature_s)
    _, s_t, _ = torch.svd(feature_t)
    sigma = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
    sigma *= 0.0001
    return sigma

    
def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()

        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 

def contras_cls(p1, p2):
    N, C = p1.shape
    cov = p1.t() @ p2

    cov_norm1 = cov / torch.sum(cov, dim=1, keepdims=True)
    cov_norm2 = cov / torch.sum(cov, dim=0, keepdims=True)
    loss = 0.5 * (torch.sum(cov_norm1) - torch.trace(cov_norm1)) / C \
        + 0.5 * (torch.sum(cov_norm2) - torch.trace(cov_norm2)) / C
    return loss


def ot_loss(proto_s, feat_tu_w, feat_tu_s, args):
    bs = feat_tu_s.shape[0]
    with torch.no_grad():
        M_st_weak = 1 - pairwise_cosine_sim(proto_s.mo_pro, feat_tu_w)  # postive distance A和原型的距离
    gamma_st_weak = ot_mapping(M_st_weak.data.cpu().numpy().astype(np.float64)) # A和原型的OT
    score_ot, pred_ot = gamma_st_weak.t().max(dim=1)
    Lm = center_loss_cls(proto_s.mo_pro, feat_tu_s, pred_ot, num_classes=args.num_classes) # B和原型的OT
    return Lm


def pairwise_cosine_sim(a, b):
    assert len(a.shape) == 2
    assert a.shape[1] == b.shape[1]
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    mat = a @ b.t()
    return mat


def ot_mapping(M):
    '''
    M: (ns, nt)
    '''
    reg1 = 1
    reg2 = 1
    ns, nt = M.shape
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt # 得到1/ns，1/nt
    gamma = ot.unbalanced.sinkhorn_stabilized_unbalanced(a, b, M, reg1, reg2)
    gamma = torch.from_numpy(gamma).cuda()
    return gamma


def center_loss_cls(centers, x, labels, num_classes=65):
    classes = torch.arange(num_classes).long().cuda()
    batch_size = x.size(0)
    centers_norm = F.normalize(centers)
    x = F.normalize(x)
    distmat = -x @ centers_norm.t() + 1

    labels = labels.unsqueeze(1).expand(batch_size, num_classes)
    mask = labels.eq(classes.expand(batch_size, num_classes))

    dist = distmat * mask.float()
    loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
    return loss




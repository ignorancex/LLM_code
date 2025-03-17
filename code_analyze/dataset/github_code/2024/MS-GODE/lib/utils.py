import os
import logging
import pickle
import time
import scipy.sparse as sp
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .gnn_models import set_model_task, get_model_scores
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from scipy.spatial import distance

def read_selected_VCell_vars(file_path):
    # convert VCell results to a dict of ndarray
    var_names = []
    with open(file_path,'r') as f:
        data = f.readlines()
        for d in data:
            var_names.append(d.replace('\n',''))
    return var_names

def print_parameters(model):
	for name, param in model.named_parameters():
		if param.requires_grad:
			print(name)
def encode_onehot(labels):
	classes = set(labels)
	classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
					enumerate(classes)}
	labels_onehot = np.array(list(map(classes_dict.get, labels)),
							 dtype=np.int32)
	return labels_onehot
def init_network_weights_lode(net, std = 0.1):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, mean=0, std=std)
			nn.init.constant_(m.bias, val=0)

class bcsr_Training():

    def __init__(self, proxy_model, beta, device, lr_proxy_model, lr_weights):
        self.proxy_model = proxy_model
        self.lr_p = lr_proxy_model
        self.lr_w = lr_weights
        self.optimizer_theta_p_model = torch.optim.SGD(self.proxy_model.parameters(), lr=self.lr_p)
        self.weight_optimizer = None
        self.device = device
        self.eta = 0.5
        self.beta = beta
        self.buffer = []
        self.identity = []

    def init_proxy_model(self):
        for m in self.proxy_model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)

    def train_inner(self, batch_dict_encoder, batch_dict_decoder, batch_dict_graph, task_id, sample_weights,
            inner_epchos, n_traj_samples=1, kl_coef=1.):
        loss = math.inf
        for _ in range(inner_epchos):
            self.optimizer_theta_p_model = torch.optim.SGD(self.proxy_model.parameters(), lr=self.lr_p)
            self.proxy_model.train()
            self.optimizer_theta_p_model.zero_grad()
            sample_weights = sample_weights.to(self.device).type(torch.float).detach()
            losses = self.proxy_model.compute_all_losses_bcsr(batch_dict_encoder, batch_dict_decoder, batch_dict_graph,
                                                            n_traj_samples=n_traj_samples, kl_coef=kl_coef)
            loss = sum([sample_weights[i] * losses[i] for i in range(len(losses))])/len(losses)
            #loss = torch.mean(sample_weights * losses)
            loss.backward()
            self.optimizer_theta_p_model.step()
            self.proxy_model.zero_grad()
        return loss

    def train_outer(self, batch_dict_encoder, batch_dict_decoder, batch_dict_graph, task_id, data_weights, topk,
            n_traj_samples=1, kl_coef=1.):
        # def train_outer(self, data, target, task_id, data_weights, topk, ref_x=None, ref_y=None):
        # batch_dict_encoder = batch_dict_encoder.to(self.device)
        # target = target.to(self.device).type(torch.long)
        sample_weights = data_weights.to(self.device)
        # X_S = data[:].to(self.device)
        # y_S = target[:].to(self.device).type(torch.long)
        return self.update_sample_weights(batch_dict_encoder, batch_dict_decoder, batch_dict_graph, task_id,
                                          sample_weights, topk, beta=self.beta, n_traj_samples=n_traj_samples,
                                          kl_coef=kl_coef)

    def update_sample_weights(self, batch_dict_encoder, batch_dict_decoder, batch_dict_graph, task_id, sample_weights,
            topk, beta, epsilon=1e-3, n_traj_samples=3, kl_coef=1.):
        z = torch.normal(0, 1, size=[topk]).cuda(self.device)

        loss_outer = self.proxy_model.compute_all_losses_bcsr(batch_dict_encoder, batch_dict_decoder, batch_dict_graph,
                                                        n_traj_samples=n_traj_samples, kl_coef=kl_coef)
        #loss_outer = train_res["all_loss"]

        # loss_outer = F.cross_entropy(self.proxy_model(input_train, task_id), target_train, reduction='none')
        topk_weights, ind = sample_weights.topk(topk)
        loss_outer_avg = sum(loss_outer)/len(loss_outer) - beta * (topk_weights + epsilon * z).sum()
        #loss_outer_avg = torch.mean(loss_outer) - beta * (topk_weights + epsilon * z).sum()

        params_w_grad = [p for p in list(self.proxy_model.parameters()) if p.requires_grad]
        d_theta = torch.autograd.grad(loss_outer_avg, params_w_grad, allow_unused=True)
        v_0_ = d_theta
        train_res_inner = self.proxy_model.compute_all_losses_bcsr(batch_dict_encoder, batch_dict_decoder, batch_dict_graph,
                                                              n_traj_samples=n_traj_samples, kl_coef=kl_coef)
        loss_inner = sum(F.softmax(sample_weights, dim=-1)[i] * train_res_inner[i] for i in range(len(train_res_inner)))/len(train_res_inner)
        #loss_inner = torch.mean(F.softmax(sample_weights, dim=-1) * train_res_inner)

        params_w_grad = [p for p in list(self.proxy_model.parameters()) if p.requires_grad]
        grads_theta = torch.autograd.grad(loss_inner, params_w_grad, allow_unused=True, create_graph=True)
        grads_theta_wo_None = []
        G_theta = []

        params_w_grad = [p for p in list(self.proxy_model.parameters()) if p.requires_grad]
        for p, g in zip(params_w_grad, grads_theta):
            if g == None:
                pass
                #G_theta.append(None)
            else:
                G_theta.append(p - self.lr_p * g)
                grads_theta_wo_None.append(g)
        v_0 = []
        for g in v_0_:
            if g != None:
                v_0.append(g)
        v_Q = v_0
        for _ in range(3):
            params_w_grad = [p for p in list(self.proxy_model.parameters()) if p.requires_grad]
            v_new = [torch.autograd.grad(G_theta[i], params_w_grad, allow_unused=True, grad_outputs=v_0[i], retain_graph=True) for i in range(len(G_theta))]
            v_0 = [torch.nan_to_num(i.detach(),nan=1.) for i in v_new]
            for i in range(len(v_0)):
                v_Q[i].add_(v_0[i].detach())

        jacobian = -torch.autograd.grad(grads_theta, sample_weights, allow_unused=True, grad_outputs=v_Q)[0]
        with torch.no_grad():
            sample_weights -= self.lr_w * jacobian

        return sample_weights, jacobian, loss_outer


def strtobool_str(val):
    """
    modified from strtobool in distutils
    Convert a string representation of truth to True or False.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def str2tasks(tasks_str, system='physics'):
    tasks = []
    if system == 'physics':
        if 's' in tasks_str or 'c' in tasks_str:
            tasks_ = tasks_str.split('__')
            for task in tasks_:
                task_ = task.split('_')
                task_[1] = int(task_[1])
                for i in range(2, 4):
                    task_[i] = float(task_[i])
                tasks.append(task_)
            return tasks
    elif system == 'NRW':
        tasks_ = tasks_str.split('__')
        for task in tasks_:
            task_ = task.split('_')
            task_[0] = int(task_[0])
            task_[1] = int(task_[1])
            task_[2] = float(task_[2])
            task_[3] = task_[3]
            tasks.append(task_)
        return tasks

    elif system == 'motion':
        tasks = [int(i) for i in tasks_str.split('_')]
        return tasks

    elif system == 'VCell':
        tasks_ = tasks_str.split('__')
        for task in tasks_:
            task_ = task.split('_')
            tasks.append(task_)
        return tasks


def captilize_title(title):
    words = title.split(' ')
    for i, w in enumerate(words):
        if w not in ['by', 'for', 'and', 'on', 'a', 'an']:
            words[i] = words[i].replace(w[0], w[0].upper(), 1)
    title = ' '.join(words)
    return title


# import imageio
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)


def get_logger(logpath, filepath, package_files=[],
        displaying=True, saving=True, debug=False, name=None):
    logger = logging.getLogger(name=name)
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
		for i, (x, y) in enumerate(inf_generator(train_loader)):
	"""
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def dump_pickle(data, filename):
    with open(filename, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


def load_pickle(filename):
    with open(filename, 'rb') as pkl_file:
        filecontent = pickle.load(pkl_file)
    return filecontent


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)


def flatten(x, dim):
    return x.reshape(x.size()[:dim] + (-1,))


def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device

def convert_sparse(graph):
	graph_sparse = sp.coo_matrix(graph)
	edge_index = np.vstack((graph_sparse.row, graph_sparse.col))
	edge_attr = graph_sparse.data
	return edge_index, edge_attr

def sample_standard_gaussian(mu, sigma):
    device = get_device(mu)

    d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()


def get_dict_template():
    return {"data": None,
            "time_setps": None,
            "mask": None
            }


def get_next_batch_new(dataloader, device):
    data_dict = dataloader.__next__()
    # device_now = data_dict.batch.device
    return data_dict.to(device)


def get_next_batch(dataloader, device):
    # Make the union of all time points and perform normalization across the whole dataset
    data_dict = dataloader.__next__()

    batch_dict = get_dict_template()

    batch_dict["data"] = data_dict["data"].to(device)
    batch_dict["time_steps"] = data_dict["time_steps"].to(device)
    batch_dict["mask"] = data_dict["mask"].to(device)

    return batch_dict


def get_ckpt_model(ckpt_path, model, device):
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")
    # Load checkpoint.
    checkpt = torch.load(ckpt_path)
    ckpt_args = checkpt['args']
    state_dict = checkpt['state_dict']
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(state_dict)
    # 3. load the new state dict
    model.load_state_dict(state_dict)
    model.to(device)


def update_learning_rate(optimizer, decay_rate=0.999, lowest=1e-3):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr


def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert (start.size() == end.size())
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res,
                             torch.linspace(start[i], end[i], n_points)), 0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res


def reverse(tensor):
    idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
    return tensor[idx]


def create_net(n_inputs, n_outputs, n_layers=1,
        n_units=100, nonlinear=nn.Tanh):
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)


def compute_loss_all_batches(args, model,
        encoder, graph, decoder, encoder_mask, graph_mask, decoder_mask,
        n_batches, device,
        n_traj_samples=1, kl_coef=1., task_id=None):
    total = {}
    total["loss"] = 0
    total["likelihood"] = 0
    total["mse"] = 0
    total["kl_first_p"] = 0
    total["std_first_p"] = 0

    n_test_batches = 0

    model.eval()
    # print("Computing loss... ")
    inferred_tasks = []
    klds = []
    # for i in tqdm(range(n_batches)):
    for i in range(n_batches):
        batch_dict_encoder = get_next_batch_new(encoder, device)
        batch_dict_graph = get_next_batch_new(graph, device)
        batch_dict_decoder = get_next_batch(decoder, device)

        t0 = time.time()

        if args.mask:
            batch_dict_enc_mask = get_next_batch_new(encoder_mask, device)
            batch_dict_g_mask = get_next_batch_new(graph_mask, device)
            batch_dict_dec_mask = get_next_batch(decoder_mask, device)
            if task_id is None:
                # no task id is given (should be in testing)
                if args.mask_select == 'one_shot_gradient':
                    # use a single mask
                    set_model_task(model, -1, verbose=False)
                    inferred_task = get_mask(model, batch_dict_enc_mask, batch_dict_g_mask, batch_dict_dec_mask,
                                             device=device,
                                             n_traj_samples=3, kl_coef=kl_coef, mseORloss=args.mask_criteria)
                    inferred_tasks.append(inferred_task)
                elif args.mask_select == 'weighted_sum':
                    # use weighted summation of masks
                    set_model_task(model, -1, verbose=False)
                    inferred_task = -1  # still use super position of masks
                    alphas = get_mask_combine(model, batch_dict_enc_mask, batch_dict_g_mask, batch_dict_dec_mask,
                                              device=device,
                                              n_traj_samples=3, kl_coef=kl_coef, mseORloss=args.mask_criteria)
                    inferred_tasks.append(alphas)
                elif args.mask_select == 'mask_prototype':
                    proto_distance = {}
                    for m in range(model.num_tasks_learned):
                        set_model_task(model, m, verbose=False)
                        proto_distance[m] = get_proto_distance(model, batch_dict_enc_mask, batch_dict_g_mask,
                                                               batch_dict_dec_mask, m)
                    inferred_task = min(proto_distance, key=proto_distance.get)
                    inferred_tasks.append(inferred_task)
                    klds.append(proto_distance)
                elif args.mask_select == 'part_recon':
                    # use a single mask
                    set_model_task(model, -1, verbose=False)
                    inferred_task = get_mask_part_recon(model, batch_dict_enc_mask, batch_dict_g_mask,
                                                        batch_dict_dec_mask,
                                                        device=device,
                                                        n_traj_samples=3, kl_coef=kl_coef, mseORloss=args.mask_criteria)
                    inferred_tasks.append(inferred_task)

            else:
                # task id is given (should be in training)
                inferred_task = task_id

            set_model_task(model, inferred_task, verbose=False)

        # print('set mask time:', time.time() - t0)
        t0 = time.time()
        with torch.no_grad():
            results = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph,
                                               n_traj_samples=n_traj_samples, kl_coef=kl_coef)
        # print('with torch.nograd inference time:',time.time()-t0)

        for key in total.keys():
            if key in results:
                var = results[key]
                if isinstance(var, torch.Tensor):
                    var = var.detach().item()
                total[key] += var

        n_test_batches += 1

        del batch_dict_encoder, batch_dict_graph, batch_dict_decoder, results

    print(f'inferred task {inferred_tasks}')
    print(f'klds {klds}')

    if n_test_batches > 0:
        for key, value in total.items():
            total[key] = total[key] / n_test_batches
    return total


def get_proto_distance(model, batch_dict_enc_mask, batch_dict_g_mask, batch_dict_dec_mask, task_id, dist_mode='proto',
        dist_metric='kld'):
    model.eval()
    model.zero_grad()
    with torch.no_grad():
        first_point_mu, first_point_std = model.encoder_z0(batch_dict_enc_mask.x, batch_dict_enc_mask.edge_attr,
                                                           batch_dict_enc_mask.edge_index, batch_dict_enc_mask.pos,
                                                           batch_dict_enc_mask.edge_same,
                                                           batch_dict_enc_mask.batch,
                                                           batch_dict_enc_mask.y)  # [num_ball,10]
    if dist_mode == 'proto':
        # dist = distance.cosine(first_point_mu.cpu().detach().numpy(),model.mu_proto[task_id].cpu())
        if dist_metric == 'cosine':
            dist = 1. - torch.nn.functional.cosine_similarity(first_point_mu.mean(0), model.mu_proto[task_id],
                                                              dim=0).detach()
        elif dist_metric == 'kld':
            dist = kl_divergence(Normal(first_point_mu.mean(0), first_point_std.mean(0)),
                                 Normal(model.mu_proto[task_id], model.std_proto[task_id]))
            dist = torch.mean(dist).detach()
    return dist


def get_mask(model, encoder, graph, decoder, device, n_traj_samples=1, kl_coef=1., mseORloss='loss'):
    model.eval()
    # set alphas
    model.zero_grad()
    model.apply(lambda m: setattr(m, "task", -1))
    alphas = (torch.ones([model.num_tasks_learned, 1, 1], device=device, requires_grad=True) / model.num_tasks)
    model.apply(lambda m: setattr(m, "alphas", alphas))
    # print("Computing loss... ")
    results = model.compute_all_losses(encoder, decoder, graph,
                                       n_traj_samples=n_traj_samples, kl_coef=kl_coef)
    loss = results[mseORloss]
    # del encoder,graph,decoder, results
    grad = torch.autograd.grad(loss, alphas)[0]
    print('grads for mask selection', grad)
    inferred_task = (-grad).squeeze().argmax()
    inferred_task = inferred_task.item()

    return inferred_task


def get_mask_part_recon(model, encoder, graph, decoder, device, n_traj_samples=1, kl_coef=1., mseORloss='loss'):
    model.eval()
    errors = []
    # test each mask and select one
    for t in range(model.num_tasks_learned):
        model.zero_grad()
        model.apply(lambda m: setattr(m, "task", t))
        results = model.compute_all_losses(encoder, decoder, graph,
                                           n_traj_samples=n_traj_samples, kl_coef=kl_coef)
        loss = results[mseORloss]
        errors.append(loss.item())
    inferred_task = np.argmin(errors)
    return inferred_task


def get_mask_combine(model, encoder, graph, decoder, device, n_traj_samples=1, kl_coef=1., mseORloss='loss',
        adaptation_steps=10, optimizer='AdamW', lr=5e-3, weight_decay=1e-3):
    # get a set of weights instead of only returning only one mask
    model.eval()
    # set alphas
    model.zero_grad()
    model.apply(lambda m: setattr(m, "task", -1))
    alphas = torch.ones([model.num_tasks_learned, 1, 1], device=device, requires_grad=True)
    optimizers = {"AdamW": optim.AdamW, "Adam": optim.Adam}
    optimizer = optimizers[optimizer]([alphas], lr=lr, weight_decay=weight_decay)

    # print("Computing loss... ")
    for ite in range(adaptation_steps):
        model.apply(lambda m: setattr(m, "alphas", (F.softmax(alphas))))
        results = model.compute_all_losses(encoder, decoder, graph,
                                           n_traj_samples=n_traj_samples, kl_coef=kl_coef)
        loss = results[mseORloss]
        # grad = torch.autograd.grad(loss, alphas)[0]
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

    return F.softmax(alphas)


def cond_cache_masks(m, ):
    if hasattr(m, "cache_masks"):
        m.cache_masks()


def cache_masks(model):
    model.apply(cond_cache_masks)


def n_tasks_learnt(m, num_tasks_learned):
    if 1 == 1:  # hasattr(m, "cache_masks"):
        m.num_tasks_learned = num_tasks_learned


def set_num_tasks_learned(model, num_tasks_learned):
    model.apply(lambda m: n_tasks_learnt(m, num_tasks_learned))



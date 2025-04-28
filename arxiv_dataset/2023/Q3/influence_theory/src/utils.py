"""This file support run_train_model, run_IF_exp, and run_MIS_exp
"""
import transformers
import torch
import torch.nn as nn
import re
import logging
from src.loss_functions import multi_loss_fn


# --------------------Model and Tokenizer-------------------------------------------------------
# Model and tokenizer downloading functions are altered from Mitchel et. al. (2022), MEND (https://github.com/eric-mitchell/mend;https://openreview.net/pdf?id=0DcZxeWfOPt)
def scr():
    scr_dir = "../src"


def get_model(tokenizer, model_name, transformers, model_class_name, model_pt, dropout, base_dir=""):
    ModelClass = getattr(transformers, model_class_name)
    logging.info(
        f"Loading model class {ModelClass} with name {model_name} from cache dir {scr()}")
    if model_class_name == "GPT2LMHeadModel":
        model = ModelClass.from_pretrained(
            model_name, pad_token_id=tokenizer.eos_token_id, cache_dir=scr())
    else:
        model = ModelClass.from_pretrained(model_name, cache_dir=scr())

    if model_pt is not None:
        logging.info(f"Loading model initialization from {model_pt}")
        # if model_class_name == "GPT2LMHeadModel":
        #     # Adding padding to tokenizer
        #     model.resize_token_embeddings(len(tokenizer))
        #     model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(
        #         0)
        state_dict = torch.load(base_dir+model_pt, map_location="cpu")

        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            logging.info(
                "Default load failed; stripping prefix and trying again.")
            state_dict = {re.sub("^model.", "", k): v for k,
                          v in state_dict.items()}

            model.load_state_dict(state_dict)

        logging.info("Loaded model initialization")

    if dropout is not None:
        n_reset = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout
                n_reset += 1

            if hasattr(m, "dropout"):  # Requires for BART, which uses F.dropout
                if isinstance(m.dropout, float):
                    m.dropout = dropout
                    n_reset += 1

            if hasattr(m, "activation_dropout"):  # Requires for BART, which uses F.dropout
                if isinstance(m.activation_dropout, float):
                    m.activation_dropout = dropout
                    n_reset += 1

        logging.info(f"Set {n_reset} dropout modules to p={dropout}")
    return model


def get_tokenizer(model_tokenizer_name, model_name, model_tokenizer_class):
    tok_name = model_tokenizer_name if model_tokenizer_name is not None else model_name
    return getattr(transformers, model_tokenizer_class).from_pretrained(tok_name, cache_dir=scr())
# ---------------------------------------------------------------------------

# Creates dictionary
def dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v
    return new_dict


# Code from https://github.com/xhan77/influence-function-analysis/blob/78d5a967aba885f690d34e88d68da8678aee41f1/bert_util.py#L336
def unflatten_to_param_dim(x, param_shape_tensor):
    tar_p = []
    ptr = 0
    for p in param_shape_tensor:
        len_p = torch.numel(p)
        tmp = x[ptr: ptr + len_p].view(p.shape)
        tar_p.append(tmp)
        ptr += len_p
    return tar_p

# Code from https://github.com/xhan77/influence-function-analysis/blob/78d5a967aba885f690d34e88d68da8678aee41f1/bert_util.py#L336
def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

# Average gradient of loss function
def avg_grad(data, model, device, param_influence, task):
    # Initialize correctly on the right device
    grad_all = [torch.zeros_like(v) for v in model.parameters()]
    count = 0
    for batch in data:
        loss = multi_loss_fn(model, batch, task)
        batch_grad = torch.autograd.grad(loss, param_influence)
        bs = len(batch)  # batch size
        grad_all = [(a*count)/(count+bs)+(bt*bs)/(count+bs)
                    for a, bt in zip(grad_all, batch_grad)]
        count += bs
    return grad_all

# Hessian vector product over one batch
def hvp_fn(model, loss, multiplier):
    grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    gnorm = 0
    for m, g in zip(multiplier, grad):
        gnorm = gnorm + torch.dot(torch.flatten(m), torch.flatten(g))
    hvp = torch.autograd.grad(gnorm, model.parameters(), create_graph=True)
    return [h.detach() for h in hvp]

# Hessian vector product average over batches
@torch.enable_grad()
def avg_hvp(data, model, multiplier, device, task='zsre', break_early=False):
    # Initialize correctly on the right device
    hvp_all = [torch.zeros_like(v) for v in model.parameters()]
    count = 0
    for i, batch in enumerate(data):
        loss = multi_loss_fn(model, batch, task)
        hvp_batch = hvp_fn(model, loss, multiplier)
        bs = len(batch)  # batch size
        hvp_all = [(a*count)/(count+bs)+(bt*bs)/(count+bs)
                   for a, bt in zip(hvp_all, hvp_batch)]
        count += bs
        if (break_early == True) & (i == len(data)//2):
            break
    return hvp_all

# Epoch loss for SGD, SVRG, and Conjugate Gradient
def epoch_loss(ihvp, train_loader, model, device, target, regularization_param=0.0, task="zsre"):
    hvp = avg_hvp(train_loader, model, ihvp, device, task=task)
    t1 = 0.5 * sum(torch.dot(a.view(-1), b.view(-1))
                   for a, b in zip(ihvp, hvp))
    t2 = sum(torch.dot(a.view(-1), b.view(-1)) for a, b in zip(target, ihvp))
    t3 = 0.5 * regularization_param * \
        sum(torch.dot(v.view(-1), v.view(-1)) for v in ihvp)
    loss = t1 - t2 + t3
    return loss

# Epoch loss for arnoldi
def arnoldi_loss(eigvecs, eigvals, target):
    # -.5 V^TU \lambda U^TV
    grad = []
    for i, g in enumerate(target):
        grad.append(g.clone().detach().to("cpu"))
    # Project gradients onto eigvecs
    proj_grad = []
    for ev in eigvecs:
        proj_grad.append(sum([torch.dot(torch.flatten(
            g.clone().detach()), torch.flatten(e)) for g, e in zip(grad, ev)]))
    # Diag of eigenvals
    inverse_hessian_diag = 1 / eigvals
    return torch.dot(gather_flat_grad(proj_grad), inverse_hessian_diag * gather_flat_grad(proj_grad))

# Create pytorch dataloader
def create_dataloader(data, batch_size):
    dataloader = torch.utils.data.DataLoader(
        dataset=data, batch_size=batch_size, shuffle=False)
    return (dataloader)

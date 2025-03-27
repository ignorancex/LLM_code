import torch
import torch.nn as nn
import copy as copy
import numpy as np

class SynFlowPruner:
    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}

    def score(self, model, dataloader, device, double_precision=False):

        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        signs = linearize(model)

        (data, _) = next(iter(dataloader))
        if double_precision:
            input_dim = list(data[0, :].shape)
            input = torch.ones([1] + input_dim, dtype=torch.float64).to(device)  # , dtype=torch.float64).to(device)
        else:
            input_dim = list(data[0, :].shape)
            input = torch.ones([1] + input_dim).to(device)  # , dtype=torch.float64).to(device)
        output = model(input)
        val = torch.sum(output)
        val.backward()

        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally.
        """
        # # Set score for masked parameters to -inf
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            print("Spar: {}, remove {} params smaller than {}".format(sparsity, k, threshold))
            print("there are {} such parameters".format((global_scores<threshold).sum()))
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)]
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))


    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise.
        """
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        if scope == 'global':
            self._global_mask(sparsity)
        if scope == 'local':
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, param in self.masked_parameters:
            shape = mask.shape
            perm = torch.randperm(mask.nelement())
            mask = mask.reshape(-1)[perm].reshape(shape)

    def invert(self):
        for v in self.scores.values():
            v.div_(v ** 2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        remaining_params, total_params = 0, 0
        for mask, _ in self.masked_parameters:
            remaining_params += mask.detach().cpu().numpy().sum()
            total_params += mask.numel()
        return remaining_params, total_params



# def synflow(model, keep_ratio, train_dataloader, device):
#     net = copy.deepcopy(model)
#
#     @torch.no_grad()
#     def linearize(model):
#         # model.double()
#         signs = {}
#         for name, param in model.state_dict().items():
#             signs[name] = torch.sign(param)
#             param.abs_()
#         return signs
#
#     @torch.no_grad()
#     def nonlinearize(model, signs):
#         # model.float()
#         for name, param in model.state_dict().items():
#             param.mul_(signs[name])
#
#     signs = linearize(net)
#
#     (data, _) = next(iter(train_dataloader))
#     input_dim = list(data[0, :].shape)
#     input = torch.ones([1] + input_dim).to(device)  # , dtype=torch.float64).to(device)
#     output = net(input)
#     torch.sum(output).backward()
#
#     scores = []
#     for layer in net.modules():
#         if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
#             scores.append(torch.abs(layer.weight.grad * layer.weight.data))
#     all_scores = torch.cat([torch.flatten(v) for v in scores])
#
#     num_params_to_keep = int(len(all_scores) * keep_ratio)
#     threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
#     acceptable_score = threshold[-1]
#
#     layer_wise_sparsities = []
#     layer_wise_mask = []
#     for s in scores:
#         mask = (s >= acceptable_score).float()
#         sparsity = float((mask==0).sum().item() / mask.numel())
#         layer_wise_sparsities.append(sparsity)
#         layer_wise_mask.append(mask.detach())
#     print(f'layer-wise sparsity is {layer_wise_sparsities}')
#
#     return layer_wise_sparsities, layer_wise_mask


def synflow_prune_loop(model, keep_ratio, dataloader, device, pruner,
                       schedule="exponential", scope="global", epochs=100,
                       reinitialize=False, train_mode=False, shuffle=False, invert=False, double_precision=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """

    sparsity = keep_ratio

    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()
    # Prune model
    for epoch in range(epochs):
        pruner.score(model, dataloader, device, double_precision)
        if schedule == 'exponential':
            sparse = sparsity ** ((epoch + 1) / epochs)
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity) * ((epoch + 1) / epochs)
        else:
            raise ValueError("Unknown schedule for synflow")
        # Invert scores
        if invert:
            pruner.invert()
        pruner.mask(sparse, scope)
        pruner.apply_mask()

    # Reainitialize weights
    if reinitialize:
        model._initialize_weights()

    # Shuffle masks
    if shuffle:
        pruner.shuffle()

    # Confirm sparsity level
    remaining_params, total_params = pruner.stats()


    if np.abs(remaining_params - total_params * sparsity) >= 5:
        print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params * sparsity))
        quit()
    else:
        print("SYNFLOW: {} prunable parameters remaining, expected {}".format(remaining_params, total_params * sparsity))

    layer_wise_sparsities = []
    layer_wise_mask = []
    for m, p in pruner.masked_parameters:
        sparsity = float((m==0).sum().item() / m.numel())
        layer_wise_sparsities.append(sparsity)
        layer_wise_mask.append(m.detach())
    print(f'layer-wise sparsity is {layer_wise_sparsities}')

    return layer_wise_sparsities, layer_wise_mask
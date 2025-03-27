import numpy as np
import torch
import torch.optim as optim
import orthogonal.ortho as ortho
from models.initializers import conv_orthogonal, delta_orthogonal, orthogonal_matrix
import math
from itertools import product

WEIGHTLR=0.1

def get_optimizer_for_config(args, params):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=WEIGHTLR) # accordingly to https://github.com/namhoonlee/spp-public/blob/32bde490f19b4c28843303f1dc2935efcd09ebc9/spp/model.py#L501
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=WEIGHTLR, weight_decay=0)
    else:
        print('Unknown optimizer: {0}'.format(args.optimizer))
        raise Exception('Unknown optimizer')
    return optimizer

class Preprocesser():
    def __call__(self):
        raise NotImplementedError()


class EmptyPreprocesser(Preprocesser):
    def __call__(self):
        pass

class ApproximateIsometryPreprocesser(Preprocesser):
    def __init__(self, model, masking, args, logger, device):
        self.net = model
        self.AI_iters = args.AI_iters
        self.args = args
        self.device = device
        self.masking = masking
        self.sigma_w = args.sigma_w
        self.logger = logger
        self.do_log = args.log_preprocessing

    def loss(self, n, p: torch.nn.Parameter):
        mask = self.masking.masks[n]
        if len(p.shape) == 2: # linear
            CW = p * mask
        elif len(p.shape) == 4: # conv
            CW = p.view((p.shape[0], -1)) * mask.view((mask.shape[0], -1)) # from lines 614 in https://github.com/namhoonlee/spp-public/blob/master/spp/model.py
        else:
            raise NotImplementedError("Unsupported weight shape")
        if CW.shape[0]<=CW.shape[1]:
            mul =  torch.matmul(CW, CW.T)
            return torch.norm(mul-self.sigma_w**2*torch.eye(mul.shape[0], device=self.device))
        else:
            mul =  torch.matmul(CW.T, CW)
            return torch.norm(mul-self.sigma_w**2*torch.eye(mul.shape[1], device=self.device))

    def get_loss_term(self):
        loss = torch.tensor(0.0).to(self.device)
        for n, p in self.net.named_parameters():
            if n in self.masking.masks:
                loss = loss + self.loss(n, p)
        return loss

    def repair_DI(self, n, p):
        optimizer = get_optimizer_for_config(self.args, [p])
        for i in range(self.AI_iters):
            optimizer.zero_grad()
            loss = self.loss(n, p)
            loss.backward()
            optimizer.step()
            if self.do_log:
                metrics = {"AI_loss":loss.detach().cpu().numpy()}
                self.logger.log(metrics)

    def __call__(self):
        params = []
        for n, p in self.net.named_parameters():
            if n in self.masking.masks:
                params.append(p)
        optimizer = get_optimizer_for_config(self.args, params)
        for i in range(self.AI_iters):
            optimizer.zero_grad()
            loss = self.get_loss_term()
            loss.backward()
            optimizer.step()
            if self.do_log:
                metrics = {"AI_loss":loss.detach().cpu().numpy()}
                self.logger.log(metrics)


class SparseOrthogonalPreprocesser(Preprocesser):
    def __init__(self, model, masking, args, device):
        self.net = model
        self.device = device
        self.masking = masking
        self.sigma_w = args.sigma_w
        self.sigma_b = args.sigma_b
        self.more_nonzeros = args.more_nonzeros

    def __call__(self):
        with torch.no_grad():
            self.sparse_orthogonal_init()

    def unfreeze_zeros(self, conv_weights: torch.Tensor, target_density):
        target_nonzeros = target_density * torch.numel(conv_weights)
        current_nonzeros = torch.count_nonzero(conv_weights)
        mask = (conv_weights != 0.0).float()

        if target_density == 1.0:
            mask[:, :, :, :] = 1.0
            return mask
        
        while current_nonzeros < target_nonzeros:
            idx = [0, 0, 0, 0]
            for i in range(4):
                idx[i] = np.random.randint(0, mask.shape[i])
            if mask[idx[0], idx[1], idx[2], idx[3]] == 0.0:
                current_nonzeros += 1
                mask[idx[0], idx[1], idx[2], idx[3]] = 1.0

        return mask
    
    def sparse_delta_orthogonal(self, conv_weights, target_density):
        n, m, ker_x, _ = conv_weights.shape
        assert ker_x != 2  # we don't expect 2 x 2 kernels

        if self.more_nonzeros:
            # allow larger density among the central kernel elements
            max_orthogonal_density = min(target_density * (ker_x ** 2), 1.0)
            orthogonal_density = min(max_orthogonal_density, math.sqrt(target_density))
        else:
            orthogonal_density = target_density

        orthogonal_matrix = ortho.orthogonal_with_density(n, m, orthogonal_density)
        conv_weights = delta_orthogonal(conv_weights, orthogonal_matrix, gain=self.sigma_w)
        mask = self.unfreeze_zeros(conv_weights, target_density)
        return mask
    
    def sparse_linear_orthogonal(self, linear_weights, target_density):
        n, m = linear_weights.shape
        with torch.no_grad():
            result = ortho.orthogonal_with_density(n, m, target_density).to(self.device) * self.sigma_w
            linear_weights[:, :] = result[:, :]
        mask = (linear_weights != 0.0).float().to(self.device)
        return mask
    
    def sparse_orthogonal_init(self):
        """
        Initialize orthogonal weights.
        Linear weights are initialized with orthogonal matrices.
        Conv weights are initialized with delta orthogonal kernels.
            Delta orthogonal kernel generation process:
                1. Generate sparse orthogonal matrix H
                2. Use it in delta initialization to generate an orthogonal sparse kernel
                3. Pick some number of zero-cells from the whole kernel and unfreeze them for the training

            If self.kernel_dens is not None then we only unfreeze zeros in kernels which are not completely
            zero (their corresponding entry in H is non-zero). We select dens(H) to be equal to (ovr_dens / kernel_dens)
            Otherwise we allow all zeros to be considered for unfreezing.
            In this case we set dens(H) = ovr_dens and unfreeze zeros until ovr_dens is achieved in the whole kernel.
        """
        for name, param in self.net.named_parameters():

            if 'bias' in name:  # any bias
                torch.nn.init.normal_(param.data, mean=0.0, std=self.sigma_b)
                continue

            if name in self.masking.masks:
                # Compute target density from the mask and initialize in the sparse orthogonal way with this density
                mask = self.masking.masks[name]
                mask_density = (torch.count_nonzero(mask) / torch.numel(mask)).item()

                if len(param.data.shape) == 2:  # assume linear
                    mask = self.sparse_linear_orthogonal(param.data, mask_density)
                    self.masking.masks[name][:, :] = mask[:, :]

                elif len(param.data.shape) == 4:  # assume convolutional
                    mask = self.sparse_delta_orthogonal(param.data, mask_density)
                    self.masking.masks[name][:, :, :, :] = mask[:, :, :, :]

                else:
                    raise NotImplementedError()


class StrcuturedSparseOrthogonalPreprocesser(Preprocesser):
    def __init__(self, model, masking, args, device):
        self.net = model
        self.device = device
        self.masking = masking
        self.sigma_w = args.sigma_w
        self.sigma_b = args.sigma_b
        self.more_nonzeros = args.more_nonzeros

    def __call__(self):
        with torch.no_grad():
            self.sparse_orthogonal_init()

    def _get_stuctured_mask(self, conv_weights, mask, in_channels, out_channels):
        if in_channels == 3:
            return torch.ones_like(conv_weights).to(self.device)
        delta_mask = torch.zeros_like(conv_weights).to(self.device)

        for i, j in product(
            range(out_channels), range(in_channels)
        ):
            delta_mask[i, j] = mask[i, j]

        return delta_mask

    def sparse_delta_orthogonal(self, conv_weights, target_density):
        n, m, ker_x, _ = conv_weights.shape
        assert ker_x != 2  # we don't expect 2 x 2 kernels

        orthogonal_density = target_density

        orthogonal_matrix = ortho.orthogonal_with_density(n, m, orthogonal_density)
        conv_weights = delta_orthogonal(conv_weights, orthogonal_matrix, gain=self.sigma_w)
        conv_mask = (orthogonal_matrix != 0.0).float()
        out_channels = conv_weights.shape[0]
        in_channels = conv_weights.shape[1]

        mask = self._get_stuctured_mask(conv_weights, conv_mask, in_channels, out_channels)
        return mask

    def sparse_linear_orthogonal(self, linear_weights, target_density):
        n, m = linear_weights.shape
        with torch.no_grad():
            result = ortho.orthogonal_with_density(n, m, target_density).to(self.device) * self.sigma_w
            linear_weights[:, :] = result[:, :]
        mask = (linear_weights != 0.0).float().to(self.device)
        return mask

    def sparse_orthogonal_init(self):
        """
        Initialize orthogonal weights.
        Linear weights are initialized with orthogonal matrices.
        Conv weights are initialized with delta orthogonal kernels.
            Delta orthogonal kernel generation process:
                1. Generate sparse orthogonal matrix H
                2. Use it in delta initialization to generate an orthogonal sparse kernel
                3. Pick some number of zero-cells from the whole kernel and unfreeze them for the training

            If self.kernel_dens is not None then we only unfreeze zeros in kernels which are not completely
            zero (their corresponding entry in H is non-zero). We select dens(H) to be equal to (ovr_dens / kernel_dens)
            Otherwise we allow all zeros to be considered for unfreezing.
            In this case we set dens(H) = ovr_dens and unfreeze zeros until ovr_dens is achieved in the whole kernel.
        """
        for name, param in self.net.named_parameters():

            if 'bias' in name:  # any bias
                torch.nn.init.normal_(param.data, mean=0.0, std=self.sigma_b)
                continue

            if name in self.masking.masks:
                # Compute target density from the mask and initialize in the sparse orthogonal way with this density
                mask = self.masking.masks[name]
                mask_density = (torch.count_nonzero(mask) / torch.numel(mask)).item()

                if len(param.data.shape) == 2:  # assume linear
                    mask = self.sparse_linear_orthogonal(param.data, 1.0)
                    self.masking.masks[name][:, :] = mask[:, :]

                elif len(param.data.shape) == 4:  # assume convolutional
                    mask = self.sparse_delta_orthogonal(param.data, mask_density)
                    self.masking.masks[name][:, :, :, :] = mask[:, :, :, :]

                else:
                    raise NotImplementedError()


class SparseKaimingUniformPreprocesser(Preprocesser):
    def __init__(self, model, masking):
        self.net = model
        self.masking = masking

    def __call__(self):
        self.fan_in_reinit()

    def fan_in_reinit(self):
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                if name not in self.masking.masks: # skip not pruned layers (one case are batch-norms)
                    continue
                if 'weight' in name: #TODO: Bias???
                    mask = self.masking.masks[name]
                    for w, mask_w in zip(param.data, mask): # w: rows if linear, 3d-kernels if convolutional
                        fan_in = torch.count_nonzero(mask_w)
                        if fan_in == 0:
                            continue
                        std = 1.0 / math.sqrt(fan_in)
                        torch.nn.init.uniform_(w, -std, std) # mask stays the same


class SAODeltaPreprocesser(Preprocesser):
    def __init__(self, model, masking, gain=1, sparsity=None, degree=None, same_mask = False, num_classes=10, activation="relu"):
        self.net = model
        self.masking = masking
        self.gain = gain
        self.sparsity = sparsity
        self.degree = degree
        self.same_mask = same_mask,
        self.num_classes = num_classes
        self.activation = activation

    def __call__(self):
        assert self.net.sao_called, "sao needs to be called before making the optimizer"

def get_weight_processer(args, model, masking, logger, num_classes, device):
    if args.weight_processer == "AI":
        return ApproximateIsometryPreprocesser(model, masking, args, logger, device)
    elif args.weight_processer == "sparse_orthogonal":
        return SparseOrthogonalPreprocesser(model, masking, args, device)
    elif args.weight_processer == "structured_sparse_orthogonal":
        return StrcuturedSparseOrthogonalPreprocesser(model, masking, args, device)
    elif args.weight_processer == "sparse_fan_in":
        return SparseKaimingUniformPreprocesser(model, masking)
    elif args.weight_processer == "sao":
        return EmptyPreprocesser()
    elif args.weight_processer == "none":
        return EmptyPreprocesser()
    else:
        raise ValueError("Unknown preprocesser")

from .modelargs import LLMModelArgs, MixConfig
from .model import LLMFeedForward

import torch
import torch.nn.functional as F
import random

from typing import List, Tuple, Optional
from transformers.activations import ACT2FN
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.distributions import Normal, Poisson
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected



def _mixtral_load_balancing_loss_func(
    gate_logits: List[torch.Tensor], num_experts: int, top_k: int, attention_mask: Optional[torch.Tensor] = None
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if isinstance(num_experts, List):
        num_experts = num_experts[0]
        
    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat(
        [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(
        concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
            batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(
        tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class MixtralRouterLoss(torch.nn.Module):
    def __init__(self, config: MixConfig) -> None:
        super().__init__()
        self.aux_loss_coef = config.router_aux_loss_coef_
        self.experts = config.num_experts_
        self.topk = config.top_k_

    def forward(self, gate_logits, attention_mask) -> torch.Tensor:
        gate_logits, _, _, _ = _goe_unpack_router_logits(gate_logits)
        return self.aux_loss_coef * _mixtral_load_balancing_loss_func(gate_logits, self.experts, self.topk, attention_mask)


def _mixtral_slice_tensor(data: torch.Tensor, slice: torch.Tensor,
                          dtype: torch.dtype, last_value: torch.Tensor = None):
    if last_value is None:
        return data[None, slice].reshape(-1, data.shape[-1]).to(dtype)
    else:
        return last_value


def _mixtral_compatible_forward(mlp: LLMFeedForward, moe_name: str, act_fn, expert_mask, hidden_states, input_dtype):
    final_expert_states = []
    for expert_idx in range(expert_mask.shape[0]):
        _, top_x = torch.where(expert_mask[expert_idx])
        lora_name = f"moe.{moe_name}.experts.{expert_idx}"
        lora_data = _mixtral_slice_tensor(hidden_states, top_x, input_dtype)
        final_expert_states.append(
            mlp._lora_forward(lora_name, act_fn, lora_data))

    return final_expert_states


class MixtralSparseMoe(torch.nn.Module):
    def __init__(self, args: LLMModelArgs, config: MixConfig, layer_ind: int) -> None:
        super().__init__()

        self.idx = layer_ind
        self.moe = "mixtral"
        self.adapter_name_: str = config.adapter_name
        self.dtype_: torch.dtype = torch.float32
        self.gate_ = torch.nn.Linear(
            args.dim_, config.num_experts_[layer_ind], bias=False, device=config.device, dtype=self.dtype_)
        self.act_ = ACT2FN[args.hidden_act_ if config.act_fn_ is None else config.act_fn_]
        self.experts_: int = config.num_experts_[layer_ind]
        self.topk_: int = config.top_k_
        self.jitter_noise_: float = config.jitter_noise_
        self.router_profile_: bool = False
        self.profiler_: List[int] = None

        self.lamb = torch.tensor(1.5, dtype=self.dtype_, requires_grad = True).to(config.device)
        self.theta = torch.tensor([self.experts_ / 2], dtype=self.dtype_, requires_grad = True).to(config.device)
        self.count = torch.zeros(self.experts_, dtype=self.dtype_).to(config.device)

    def _profiling(self,
                   batch_size: int,
                   sequence_length: int,
                   selected_experts: torch.Tensor) -> None:
        if not self.router_profile_:
            return

        router_statistic_ = list(0 for _ in range(self.experts_))
        for selected in selected_experts.tolist():
            for idx in selected:
                router_statistic_[idx] += 1

        if self.profiler_ is None:
            self.profiler_ = list(0 for _ in range(self.experts_))
            for idx in range(self.experts_):
                self.profiler_[idx] = (
                    router_statistic_[idx] / batch_size) / sequence_length
        else:
            for idx in range(self.experts_):
                pressure = (
                    router_statistic_[idx] / batch_size) / sequence_length
                self.profiler_[idx] = (self.profiler_[idx] + pressure) / 2

    def forward(self, mlp: LLMFeedForward, hidden_states: torch.Tensor) -> Tuple: # mlp:LlamaMLP
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        if self.jitter_noise_ > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise_, 1.0 + self.jitter_noise_)

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.view(-1, hidden_dim).to(self.dtype_)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate_(hidden_states)

        router_logits = F.softmax(router_logits, dim=1, dtype=self.dtype_)
        """if self.idx == 13 or self.idx == 27:
            print(f"idx:{self.idx},logits:{torch.sort(router_logits,descending=True)[0][10]}")
            input()"""
        routing_weights, selected_experts = torch.topk(
            router_logits, self.topk_, dim=-1)

        countt = torch.zeros_like(router_logits).to(input_dtype)
        countt = countt.scatter_add(-1, selected_experts, routing_weights.to(input_dtype))
        countt = torch.sum(countt, dim = 0)
        now_count = self.count.detach() + countt * self.topk_
        self.count = now_count.detach()

        self._profiling(batch_size, sequence_length, selected_experts)

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=self.dtype_, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.experts_).permute(2, 1, 0)
        # Perform the computation on each expert
        if hasattr(mlp, "_mixlora_forward"):
            expert_states = mlp._mixlora_forward(
                self.adapter_name_, self.act_, expert_mask, hidden_states, input_dtype)
        else:
            expert_states = _mixtral_compatible_forward(
                mlp, self.adapter_name_, self.act_, expert_mask, hidden_states, input_dtype)
        # Unpack
        for expert_idx in range(self.experts_):
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_hidden_states = expert_states[expert_idx] * \
                routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(self.dtype_))

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim).to(input_dtype)

        loss_component = torch.cat([router_logits, now_count.expand(1, self.experts_).to(self.dtype_), \
                    self.lamb.expand(1, self.experts_), self.theta.expand(1, self.experts_)], dim = 0)
        return final_hidden_states, loss_component
        return final_hidden_states, router_logits


def _switch_router_z_loss_func(router_logits: torch.Tensor) -> float:
    num_groups, tokens_per_group, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / (num_groups * tokens_per_group)

def _switch_recompute_expert_indices(router_probs: torch.Tensor, num_experts: int, expert_capacity: int) -> torch.Tensor:

    expert_index = torch.argmax(router_probs, dim=-1)
    expert_index = torch.nn.functional.one_hot(
        expert_index, num_classes=num_experts)

    # Mask tokens outside expert capacity. Sum over each sequence
    token_priority = torch.cumsum(expert_index, dim=-2)
    # mask if the token routed to to the expert will overflow
    expert_capacity_mask = token_priority <= expert_capacity
    expert_index = expert_index * expert_capacity_mask
    expert_index = torch.argmax(expert_index, dim=-1)

    return expert_index

def _switch_load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    num_experts = router_probs.shape[-1]

    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)

    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)

    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)

    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)


def _switch_unpack_router_logits(router_outputs):
    total_router_logits = []
    for router_logits in router_outputs:
        if len(router_logits.shape) > 1:
            total_router_logits.append(router_logits)
    return torch.cat(total_router_logits, dim=1)

class SwitchRouterLoss(torch.nn.Module):
    def __init__(self, config: MixConfig) -> None:
        super().__init__()
        if isinstance(config.num_experts_, List):
            self.experts = config.num_experts_[0]
        else:
            self.experts = config.num_experts_
        self.expert_capacity_ = config.expert_capacity_
        self.z_loss_coef = config.router_z_loss_coef_
        self.aux_loss_coef = config.router_aux_loss_coef_

    def forward(self, router_outputs: List[Tuple], attention_mask) -> torch.Tensor:
        router_logits = _switch_unpack_router_logits(
            router_outputs)
        z_loss = _switch_router_z_loss_func(router_logits)
        router_probs = F.softmax(router_logits, dim=-1)
        # recompute expert indexes due to m-LoRA constraints
        expert_indexes = _switch_recompute_expert_indices(
            router_probs, self.experts, self.expert_capacity_)
        aux_loss = _switch_load_balancing_loss_func(
            router_probs, expert_indexes)
        return self.z_loss_coef * z_loss + self.aux_loss_coef * aux_loss


class SwitchSparseMoe(torch.nn.Module):
    def __init__(self, args: LLMModelArgs, config: MixConfig, layer_ind: int) -> None:
        super().__init__()

        self.moe = "switch"
        self.adapter_name_: str = config.adapter_name
        self.dtype_: torch.dtype = torch.float32
        self.gate_ = torch.nn.Linear(
            args.dim_, config.num_experts_[layer_ind], bias=False, device=config.device, dtype=self.dtype_)
        self.act_ = ACT2FN[args.hidden_act_ if config.act_fn_ is None else config.act_fn_]
        self.experts_: int = config.num_experts_[layer_ind]
        self.dropout_ = torch.nn.Dropout(
            config.ffn_dropout_) if config.ffn_dropout_ > 0 else torch.nn.Identity()
        self.expert_capacity_: int = config.expert_capacity_
        self.jitter_noise_: float = config.jitter_noise_
        self.router_profile_: bool = False
        self.profiler_: List[int] = None

    def _profiling(self,
                   batch_size: int,
                   sequence_length: int,
                   router_mask: torch.Tensor) -> None:
        if not self.router_profile_:
            return

        selected_experts = torch.argmax(router_mask, dim=-1)

        router_statistic_ = list(0 for _ in range(self.experts_))
        for selected in selected_experts.tolist():
            for idx in selected:
                router_statistic_[idx] += 1

        if self.profiler_ is None:
            self.profiler_ = list(0 for _ in range(self.experts_))
            for idx in range(self.experts_):
                self.profiler_[idx] = (
                    router_statistic_[idx] / batch_size) / sequence_length
        else:
            for idx in range(self.experts_):
                pressure = (
                    router_statistic_[idx] / batch_size) / sequence_length
                self.profiler_[idx] = (self.profiler_[idx] + pressure) / 2

    def route(self, hidden_states: torch.Tensor) -> Tuple:
        if self.jitter_noise_ > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise_, 1.0 + self.jitter_noise_)

        # Apply Softmax
        router_logits = self.gate_(hidden_states)
        router_probs = F.softmax(
            router_logits, dim=-1, dtype=self.dtype_)

        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(
            expert_index, num_classes=self.experts_)

        # Mask tokens outside expert capacity. Sum over each sequence
        token_priority = torch.cumsum(expert_index, dim=-2)
        # mask if the token routed to to the expert will overflow
        expert_capacity_mask = token_priority <= self.expert_capacity_
        expert_index = expert_index * expert_capacity_mask

        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits

    def forward(self, mlp: LLMFeedForward, hidden_states: torch.Tensor) -> Tuple:
        batch_size, sequence_length, _ = hidden_states.shape

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype_)

        router_mask, router_probs, router_logits = self.route(hidden_states)
        self._profiling(batch_size, sequence_length, router_mask)

        next_states = hidden_states.clone()
        for expert_idx in range(self.experts_):
            token_indices = router_mask[:, :, expert_idx].bool()
            lora_name = f"moe.{self.adapter_name_}.experts.{expert_idx}"
            next_states[token_indices] = mlp._lora_forward(
                lora_name, self.act_, hidden_states[token_indices].to(input_dtype)).to(next_states.dtype)

        hidden_states = self.dropout_(
            router_probs * next_states).to(input_dtype)

        return hidden_states, router_logits

def _GoE_route_loss_func(alloc: torch.Tensor, lamb: torch.Tensor):

    eps = 1e-16
    n = alloc.shape[-1]
    lamb = torch.clamp(lamb, min = 1, max = 2)
    s_alloc, _ = torch.sort(alloc, dim = -1, descending=True)
    poisson_lamb = Poisson(rate = lamb)
    pois_probs = poisson_lamb.log_prob(torch.arange(1, n + 1, device = alloc.device).float().expand(lamb.shape[0], n))
    pois_probs = torch.exp(pois_probs)
    if n > 32:
        pois_probs[..., 32:] = pois_probs[..., 31:32]
    pois_probs = pois_probs / torch.sum(pois_probs, dim = -1, keepdim=True)
    pois_probs = pois_probs.unsqueeze(1)
    temp = torch.sum(pois_probs * torch.abs(torch.log(pois_probs / (s_alloc + eps) )), dim = -1) # ...,n_exp
    loss = torch.sum(temp) / (temp.numel() + eps)
    return loss

def _GoE_count_loss_func(count: torch.Tensor, miu, theta):
    eps = 1e-8
    n = count.shape[-1]
    normal = Normal(miu, theta)
    norm_prob = normal.log_prob(torch.arange(0, n, device = theta.device).float())
    norm_prob = torch.exp(norm_prob)
    norm_prob = norm_prob / torch.sum(norm_prob)
    s_count, _ = torch.sort(count, dim = -1, descending = True)
    s_norm_prob, _ = torch.sort(norm_prob, descending = True)
    temp = s_count / torch.sum(s_count, dim = -1, keepdim = True)
    loss = torch.sum(s_norm_prob * torch.abs(torch.log(s_norm_prob / (temp + torch.tensor(eps)))), dim = -1)
    loss = torch.sum(loss) / loss.numel()

    return loss

def _goe_unpack_router_logits(router_outputs):
    total_router_logits = []
    total_counts = []
    total_lambdas = []
    total_thetas = []
    for router_output in router_outputs:
        router_logits, counts, lambdas, thetas = router_output[:-3], router_output[-3], router_output[-2], router_output[-1]
        total_router_logits.append(router_logits)
        total_counts.append(counts)
        total_lambdas.append(lambdas)
        total_thetas.append(thetas)
    return torch.stack(total_router_logits, dim = 0), torch.stack(total_counts, dim = 0), \
           torch.stack(total_lambdas, dim = 0), torch.stack(total_thetas, dim = 0)

class GraphRouterLoss(torch.nn.Module):
    def __init__(self, config: MixConfig) -> None:
        super().__init__()
        self.route_loss_coef = config.router_route_loss_coef_
        self.count_loss_coef = config.router_count_loss_coef_
        if isinstance(config.num_experts_, List):
            self.experts = config.num_experts_[0]
        else:
            self.experts = config.num_experts_
        self.miu = self.experts / 2
        self.balance_strategy_ = config.balance_strategy_

        self.topk = config.top_k_

    def forward(self, router_outputs, attention_mask) -> torch.Tensor:
        router_logits, count, lamb, theta = _goe_unpack_router_logits(router_outputs)
        if self.balance_strategy_ == "goe":
            count_loss = _GoE_count_loss_func(count, self.miu, theta)
            route_loss = _GoE_route_loss_func(router_logits, lamb)
            print(f"route:{self.route_loss_coef * route_loss}, count:{self.count_loss_coef * count_loss}")
            return self.route_loss_coef * route_loss + self.count_loss_coef * count_loss
        elif self.balance_strategy_ == "mixtral":
            return self.route_loss_coef * _mixtral_load_balancing_loss_func(router_logits, self.experts, self.topk, attention_mask)
class GoEgate(torch.nn.Module):
    def __init__(self, dim, n_exp, n_layers, thresholds, dim_gcn, nl, device):
        super().__init__()
        self.dtype_: torch.dtype = torch.float32
        self.device_ = device

        self.n_layers = n_layers
        self.n_exp = n_exp
        self.convs = torch.nn.ModuleList()
        self.X = torch.empty(n_exp, dim, device = self.device_, dtype = self.dtype_, requires_grad = False)
        self.thresholds = thresholds

        for _ in range(n_layers):
            self.convs.append(GCNConv(dim_gcn, dim_gcn, bias=False))
        self.mlp = torch.nn.Linear(dim, dim_gcn, bias = False, device = self.device_, dtype = self.dtype_)
        self.mlp_struct = torch.nn.Linear(dim, dim_gcn, bias = False, device = self.device_, dtype = self.dtype_)
        self.proj = torch.nn.Linear(dim_gcn, 1, bias = False, device = self.device_, dtype = self.dtype_) # 分类映射

        self.nl = nl
        self.dim = dim
        self.edge_index = None

    def get_edges(self, X, thresholds = None):
        
        X = self.nl(self.mlp_struct(self.X))
        norm_X = F.normalize(X, p = 2, dim = 1)
        sim = torch.matmul(norm_X, norm_X.T)
        y = 1.0
        perc = 0.0
        mask = None
        if thresholds == None:
            thresholds = self.thresholds
        while perc < thresholds:
            mask = torch.abs(sim) >= y
            perc = torch.sum(mask).item() / (self.n_exp * self.n_exp)
            y -= 0.01

        edges = torch.nonzero(mask, as_tuple = False).to(X.device)

        new_edge = torch.cat((torch.arange(self.n_exp).unsqueeze(0), torch.full((1 ,self.n_exp), self.n_exp)), dim = 0).to(X.device)
        edges =  torch.cat((edges.T, new_edge), dim = 1)
        new_edge = torch.cat((torch.full((1 ,self.n_exp), self.n_exp), torch.arange(self.n_exp).unsqueeze(0)), dim = 0).to(X.device)
        edges =  torch.cat((edges, new_edge), dim = 1)
        
        edges = edges.to(torch.long)
        nw_edge = edges
        nw = 0
        ans = [0]
        for i in range(1750):
            temp = SparseTensor(row = nw_edge[0], col = nw_edge[1], sparse_sizes = ((i + 1) * (self.n_exp + 1), (i + 1) * (self.n_exp + 1)))
            ans.append(temp)
            nw += (self.n_exp + 1)
            nw_edge = torch.cat((nw_edge, edges + nw), dim = 1) 

        return ans
    
    def forward(self, x: torch.Tensor):
        x = self.nl(self.mlp(x)) # linear project
        exp = self.nl(self.mlp_struct(self.X))
        results = []
        n_loop, dim = x.shape
        
        """edge_ids = []##
        x_input = []##
        N = self.n_exp + 1##
        nw = 0##
        for i in range(n_loop):
            edge_ids.append(self.edge_index + nw)##
            nw += N##
            x_input.append(torch.cat((exp, x[i].unsqueeze(0)), dim = 0))##
        
        edge_id = torch.cat(edge_ids, dim = 1)##
        node_x = torch.cat(x_input, dim = 0)##
        edge_indexs = SparseTensor(row = edge_id[0], col = edge_id[1], sparse_sizes = (n_loop * N, n_loop * N))##
        """
        exp_expand = exp.unsqueeze(0).expand(n_loop, self.n_exp, dim)
        x_expand = x.unsqueeze(1)
        node_x = torch.cat((exp_expand, x_expand), dim = 1)
        node_x = node_x.view(-1, dim)
        edge_indexs = self.edge_index[n_loop]


        y ,edge = node_x, edge_indexs
        for conv in self.convs:
            y = self.nl(conv(y, edge))

        y = self.proj(y).squeeze()
        y = y.reshape(-1, self.n_exp + 1)
        y = y[:, :-1]
        results.append(y)
        
        result = torch.cat(results, dim = 0)
        return result


class GraphSparseMoe(torch.nn.Module):
    def __init__(self, args: LLMModelArgs, config: MixConfig, layer_ind: int) -> None:
        super().__init__()
        
        self.idx = layer_ind
        self.moe = "goe"
        self.adapter_name_: str = config.adapter_name
        self.dtype_: torch.dtype = torch.float32
        if isinstance(config.num_experts_, List):
            self.experts_ = config.num_experts_[0]
        else:
            self.experts_: int = config.num_experts_
        self.topk_: int = config.top_k_
        self.act_ = ACT2FN[args.hidden_act_ if config.act_fn_ is None else config.act_fn_]
        self.gate_ = GoEgate(args.dim_, self.experts_, config.num_gcnlayer, \
                                config.edges_thresholds, config.dim_gcn, self.act_, \
                                device=config.device)
        self.lamb = torch.tensor(1.5, dtype=self.dtype_, requires_grad = True, device=config.device)
        self.theta = torch.tensor([self.experts_ / 2], dtype=self.dtype_, requires_grad = True, device=config.device)
        self.count = torch.zeros(self.experts_, dtype=self.dtype_, device=config.device)

        self.router_profile_: bool = False
        self.profiler_: List[int] = None

    def forward(self, mlp: LLMFeedForward, hidden_states: torch.Tensor) -> Tuple:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        hidden_states = hidden_states.to(self.dtype_)

        # Apply Softmax
        router_logits = self.gate_(hidden_states)
        router_logits = F.softmax(router_logits, dim=-1)
        """if self.idx == 13 or self.idx == 27:
            print(f"idx:{self.idx},logits:{torch.sort(router_logits,descending=True)[0][20]}")
            input()"""
        routing_weights, selected_experts = torch.topk(
            router_logits.to(input_dtype), self.topk_, dim=-1) # A, B, k 
        countt = torch.zeros_like(router_logits).to(input_dtype)
        countt = countt.scatter_add(-1, selected_experts, routing_weights.to(input_dtype))
        countt = torch.sum(countt, dim = 0)
        now_count = self.count.detach() + countt * self.topk_
        self.count = now_count.detach()
        """if self.idx == 27 or self.idx == 13 or self.idx == 0:
            print(f"count:{self.count}")
            input()"""
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=self.dtype_, device=hidden_states.device
        )
        
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.experts_).permute(2, 1, 0)
        # Perform the computation on each expert
        expert_states = mlp._mixlora_forward(
            self.adapter_name_, self.act_, expert_mask, hidden_states, input_dtype)
        # Unpack
        for expert_idx in range(self.experts_):
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_hidden_states = expert_states[expert_idx] * routing_weights[top_x, idx, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(self.dtype_))
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim).to(input_dtype)
        loss_component = torch.cat([router_logits, now_count.expand(1, self.experts_).to(self.dtype_), \
                                    self.lamb.expand(1, self.experts_), self.theta.expand(1, self.experts_)], dim = 0)
        return final_hidden_states, loss_component


router_loss_dict = {
    "mixtral": MixtralRouterLoss,
    "switch": SwitchRouterLoss,
    "goe": GraphRouterLoss
}


def router_loss_factory(config: MixConfig) -> torch.nn.Module:
    if config.balance_strategy_ not in router_loss_dict:
        raise ValueError(
            f"Unknown routing strategy {config.balance_strategy_}")
    if config.router_loss_:
        return router_loss_dict[config.balance_strategy_](config)
    else:
        return None


moe_layer_dict = {
    "mixtral": MixtralSparseMoe,
    "switch": SwitchSparseMoe,
    "goe": GraphSparseMoe
}


def moe_layer_factory(args: LLMModelArgs, config: MixConfig, index: int) -> torch.nn.Module:
    if config.routing_strategy_ not in router_loss_dict:
        raise ValueError(
            f"Unknown routing strategy {config.routing_strategy_}")
    return moe_layer_dict[config.routing_strategy_](args, config, index)

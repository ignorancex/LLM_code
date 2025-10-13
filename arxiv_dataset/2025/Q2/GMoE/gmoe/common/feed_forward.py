from .modelargs import MixConfig, LLMModelArgs, MultiLoraBatchData
from .lora_linear import get_range_tensor, Linear
from .mix_lora import moe_layer_factory
#from ..models.modeling_llama import LlamaMLP

from typing import Tuple, Dict, List, Optional
import torch


class FeedForward(torch.nn.Module):
    def __init__(self, mlp) -> None:  # mlp是LlamaMLP
        super().__init__()
        self.mlp_ = mlp
        # mix of experts
        self.moes_: torch.ModuleDict = {}

    def state_dict(self) -> Dict[str, Linear]:
        return self.mlp_.state_dict()

    def forward(self, data: torch.Tensor, input_args: MultiLoraBatchData) -> Tuple[torch.Tensor, List]:
        if len(self.moes_) == 0:
            return self.mlp_._batch_forward(data, input_args), []
        else:
            return self._mixlora_forward(data, input_args)

    # MixLoRA
    def init_moe_weight(self, args: LLMModelArgs, config: MixConfig, index = None, gate = None):
        self.moes_[config.adapter_name] = moe_layer_factory(args, config, index)
        if config.routing_strategy_ == "goe":
            if gate is None:
                torch.nn.init.xavier_uniform_(
                    self.moes_[config.adapter_name].gate_.X)
                torch.nn.init.xavier_uniform_(
                    self.moes_[config.adapter_name].gate_.mlp.weight, gain = 100)
                torch.nn.init.xavier_uniform_(
                    self.moes_[config.adapter_name].gate_.mlp_struct.weight)
                torch.nn.init.xavier_uniform_(
                    self.moes_[config.adapter_name].gate_.proj.weight)
                for conv in self.moes_[config.adapter_name].gate_.convs:
                    torch.nn.init.xavier_uniform_(conv.lin.weight)
                if isinstance(config.edges_thresholds, float):
                    self.moes_[config.adapter_name].gate_.edge_index = self.moes_[config.adapter_name].gate_.\
                        get_edges(self.moes_[config.adapter_name].gate_.X, thresholds = config.edges_thresholds)
                else:
                    self.moes_[config.adapter_name].gate_.edge_index = self.moes_[config.adapter_name].gate_.\
                        get_edges(self.moes_[config.adapter_name].gate_.X, thresholds = config.edges_thresholds[int(index // 8)]).to(args.device_)
                self.moes_[config.adapter_name].gate_.convs.to(args.device_)
                self.moes_[config.adapter_name].count.zero_()
            else:
                with torch.no_grad():
                    weight, gate_name = gate
                    self.moes_[config.adapter_name].gate_.X.copy_(weight[gate_name + "X"])
                    self.moes_[config.adapter_name].gate_.mlp.weight.copy_(weight[gate_name + "mlp.weight"])
                    self.moes_[config.adapter_name].gate_.mlp_struct.weight.copy_(weight[gate_name + "mlp_struct.weight"])
                    self.moes_[config.adapter_name].gate_.proj.weight.copy_(weight[gate_name + "proj.weight"])
                    if isinstance(config.edges_thresholds, float):
                        self.moes_[config.adapter_name].gate_.edge_index = self.moes_[config.adapter_name].gate_.\
                            get_edges(weight[gate_name + "X"], thresholds = config.edges_thresholds)
                    else:
                        self.moes_[config.adapter_name].gate_.edge_index = self.moes_[config.adapter_name].gate_.\
                            get_edges(weight[gate_name + "X"], thresholds = config.edges_thresholds[int(index // 8)]).to(args.device_)
                    for i in range(config.num_gcnlayer):
                        self.moes_[config.adapter_name].gate_.convs[i].lin.weight.copy_(
                            weight[gate_name + f"convs.{i}.lin.weight"]).to(args.device_)
                    self.moes_[config.adapter_name].gate_.convs.to(args.device_)
                    self.moes_[config.adapter_name].theta.copy_(weight[gate_name + "theta"])
                    self.moes_[config.adapter_name].lamb.copy_(weight[gate_name + "lamb"])
        else:
            if gate is None:
                torch.nn.init.normal_(
                    self.moes_[config.adapter_name].gate_.weight, mean=0.0, std=config.router_init_range_)
            else:
                with torch.no_grad():
                    self.moes_[config.adapter_name].gate_.weight.copy_(gate)

    def _mixlora_forward(self, data: torch.Tensor, input_args: MultiLoraBatchData):
        final_hidden_states = torch.zeros_like(data)

        if input_args.output_router_logits_:
            router_logits = [None for _ in range(
                len(input_args.lora_batch_data_config_))]
        else:
            router_logits = []

        lora_range = get_range_tensor(data.device, data.shape[0])
        for idx, lora_config in enumerate(input_args.lora_batch_data_config_):
            moe_name = lora_config.adapter_name_
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            if moe_name in self.moes_:
                current_hidden_states, current_router_outputs = self.moes_[
                    moe_name].forward(self.mlp_, data[start_idx:end_idx])

                if input_args.output_router_logits_ and current_router_outputs is not None:
                    router_logits[idx] = current_router_outputs
            else:
                current_hidden_states = self.mlp_._lora_forward(
                    moe_name, self.mlp_.act_, data[start_idx:end_idx])

            final_hidden_states.index_add_(
                0, lora_range[start_idx:end_idx], current_hidden_states)

        return final_hidden_states, router_logits

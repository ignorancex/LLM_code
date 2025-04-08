# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, NLLLoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING, LlamaPreTrainedModel, LlamaModel, LLAMA_START_DOCSTRING
import pdb
import json
from peft.peft_model import PeftModel,  PeftConfig, _get_batch_size, PeftType
import warnings
from transformers import AutoTokenizer
import gc
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer
from accelerate import Accelerator
from scipy.spatial.distance import euclidean, cityblock, cosine, jaccard
import numpy as np

from utils.helper import clear_gpu_memory, clear_memory

clear_gpu_memory()
clear_memory()

llama3_path = "/root/autodl-tmp/cache_dir/LLM-Research/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(llama3_path)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "right"
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


class NLFTLlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, threshold, gama):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.threshold = threshold
        self.gama = gama
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            weight: Optional[torch.Tensor] = None,
            polarity: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            input_ids_wrap=None,
            attention_mask_wrap=None,
            labels_wrap=None,
            input_ids_correct=None,
            attention_mask_correct=None,
            labels_correct=None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        if input_ids_wrap is not None:
            input_ids = torch.cat([input_ids_wrap, input_ids, input_ids_correct], dim=0)
            attention_mask = torch.cat([attention_mask_wrap, attention_mask, attention_mask_correct], dim=0)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        probs = torch.softmax(logits,dim=2)
        batch_size3, seq_length, hidden_size = probs.size()
        batch_size = batch_size3 // 3
        # hidden_states_base = hidden_states[0]
        hidden_states_base = hidden_states[batch_size:2 * batch_size]

        loss = None

        if labels is not None:
            unlike_mask = polarity.eq(0).view(-1).to(probs.device)

            loss = 0
            shift_probs_pos = probs[:batch_size][..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = NLLLoss()
            shift_probs_pos = shift_probs_pos.view(-1, self.config.vocab_size)
            shift_logits = torch.log(shift_probs_pos)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            if unlike_mask.any():
                loss_unlike = self.unlikelihood(
                    probs[batch_size:2 * batch_size][unlike_mask], # base
                    probs[:batch_size][unlike_mask], # wrap
                    probs[2 * batch_size:][unlike_mask], # correct
                    labels[unlike_mask],
                    labels_wrap[unlike_mask],
                    labels_correct[unlike_mask],
                    hidden_states_base
                )

                loss = (loss_unlike + loss) * 10
            else:
                loss_correct = self.correct_loss(
                    probs[:batch_size],
                    probs[2 * batch_size:],
                    labels,
                    labels_correct,
                    hidden_states_base
                )
                loss = (loss_correct + loss) * 10
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # def l1_regularization(self, model, lambda_l1):
    #     l1_loss = 0
    #     for param in model.parameters():
    #         l1_loss += torch.norm(param, 1)  # L1 norm
    #     return lambda_l1 * l1_loss
    #
    # def l2_regularization(self, model, lambda_l2):
    #     l2_loss = 0
    #     for param in model.parameters():
    #         l2_loss += torch.norm(param, 2) ** 2  # L2 norm (squared)
    #     return lambda_l2 * l2_loss

    def measure_hidden_state_distance(self, hidden_state1, hidden_state2, method='cosine'):
        """
        度量两个隐藏状态之间的距离。

        参数:
        hidden_state1 (torch.Tensor): 第一个隐藏状态
        hidden_state2 (torch.Tensor): 第二个隐藏状态
        method (str): 距离度量方法，可选值为 'euclidean', 'manhattan', 'cosine', 'jaccard'

        返回:
        float: 两个隐藏状态之间的距离
        """
        # 将隐藏状态展平，以便计算距离
        flat_hidden_state1 = hidden_state1.flatten()
        flat_hidden_state2 = hidden_state2.flatten()

        if method == 'euclidean':
            return torch.norm(flat_hidden_state1 - flat_hidden_state2).item()
        elif method == 'manhattan':
            return torch.sum(torch.abs(flat_hidden_state1 - flat_hidden_state2)).item()
        elif method == 'cosine':
            return 1 - torch.nn.functional.cosine_similarity(flat_hidden_state1.unsqueeze(0),
                                                             flat_hidden_state2.unsqueeze(0)).item()
        elif method == 'jaccard':
            # 对于杰卡德距离，确保使用二进制向量
            binary_hidden_state1 = (flat_hidden_state1 > 0).float()
            binary_hidden_state2 = (flat_hidden_state2 > 0).float()
            intersection = torch.sum(binary_hidden_state1 * binary_hidden_state2)
            union = torch.sum(binary_hidden_state1) + torch.sum(binary_hidden_state2) - intersection
            return 1 - (intersection / union).item()
        else:
            raise ValueError("Unknown method. Available methods: 'euclidean', 'manhattan', 'cosine', 'jaccard'")

    def correct_loss(self, probs_base, probs_correct, labels_base, labels_correct, hidden_states_base):

        labels_base = labels_base.to(probs_base.device)
        labels_correct = labels_correct.to(probs_base.device)

        shift_probs_base = probs_base[..., :-1, :].contiguous()

        shift_probs_correct = probs_correct[..., :-1, :].contiguous()

        shift_labels_base = labels_base[..., 1:].contiguous()
        shift_labels_correct = labels_correct[..., 1:].contiguous()
        shift_hidden_states_base = hidden_states_base[..., :-1, :].contiguous()
        valid_indices_base = shift_labels_base != -100
        valid_indices_correct = shift_labels_correct != -100

        batch_size, seq_length, hidden_size = shift_probs_base.size()
        device = shift_probs_base.device

        label_clamped_base = torch.clamp(shift_labels_base, min=0, max=hidden_size - 1)
        label_clamped_correct = torch.clamp(shift_labels_correct, min=0, max=hidden_size - 1)
        rows, cols = torch.meshgrid(torch.arange(batch_size, device=device), torch.arange(seq_length, device=device))

        probs_out_base = shift_probs_base[rows, cols, label_clamped_base]
        probs_out_correct = shift_probs_correct[rows, cols, label_clamped_correct]

        valid_prob_base = probs_out_base[valid_indices_base]
        valid_prob_correct = probs_out_correct[valid_indices_correct]
        valid_prob_base[valid_prob_base == 0] += 1e-5  # avoid 0
        correct_threshold = 0.95
        ratio = (valid_prob_correct / correct_threshold)
        correct_indices = (ratio > 1).detach()
        hidden_states_out_base = shift_hidden_states_base[valid_indices_base]

        all_indices = ratio >= 0

        clus_indices = self.token_cluster(hidden_states_out_base, correct_indices).detach()
        scale = (1 + (((valid_prob_correct - correct_threshold) / (1 - correct_threshold)) ** 5)).detach()
        scale2 = ((valid_prob_correct / correct_threshold) ** 0.3).detach()
        scale3 = ((valid_prob_correct / correct_threshold) ** 0.6).detach()

        valid_lprob_base = torch.log(valid_prob_base)
        scale3[all_indices] = scale3[all_indices]
        scale3[clus_indices] = scale2[clus_indices]
        scale3[correct_indices] = scale[correct_indices]
        valid_lprob_base = scale3 * valid_lprob_base
        loss_correct = -torch.sum(valid_lprob_base) / valid_lprob_base.size(0)

        return loss_correct

    def token_cluster(self, hidden_states_base, indices, x=0.4):
        if not isinstance(hidden_states_base, torch.Tensor):
            raise TypeError("hidden_states_base should be a torch.Tensor")
        length = indices.size(0)
        indices = torch.nonzero(indices).squeeze()

        data_length = hidden_states_base.size(0)

        new_indices = set()
        indices = indices.long()
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)

        for idx in indices:
            idx = idx.item()
            current_center = hidden_states_base[idx]

            # 向左聚集
            left_idx = idx - 1
            while left_idx >= 0:
                similarity = self.measure_hidden_state_distance(current_center, hidden_states_base[left_idx])
                if similarity > x:
                    new_indices.add(left_idx)
                    current_center = (current_center + hidden_states_base[left_idx]) / 2
                    left_idx -= 1
                else:
                    break

            # 向右聚集
            right_idx = idx + 1
            while right_idx < data_length:
                similarity = self.measure_hidden_state_distance(current_center, hidden_states_base[right_idx])
                if similarity > x:
                    new_indices.add(right_idx)
                    current_center = (current_center + hidden_states_base[right_idx]) / 2
                    right_idx += 1
                else:
                    break

            # 添加起始点
            new_indices.add(idx)
        out_indices = torch.tensor(list(new_indices), dtype=torch.long)

        bool_tensor = torch.zeros(length, dtype=torch.bool)
        bool_tensor[out_indices] = True

        return bool_tensor

    def unlikelihood(self, probs_base, probs_wrap, probs_correct, labels_base, labels_wrap, labels_correct,
                     hidden_states_base):

        labels_wrap = labels_wrap.to(probs_base.device)
        labels_base = labels_base.to(probs_base.device)
        labels_correct = labels_correct.to(probs_base.device)

        shift_probs_base = probs_base[..., :-1, :].contiguous()
        shift_probs_wrap = probs_wrap[..., :-1, :].contiguous()
        shift_probs_correct = probs_correct[..., :-1, :].contiguous()
        shift_labels_wrap = labels_wrap[..., 1:].contiguous()
        shift_labels_base = labels_base[..., 1:].contiguous()
        shift_labels_correct = labels_correct[..., 1:].contiguous()

        shift_hidden_states_base = hidden_states_base[..., :-1, :].contiguous()

        valid_indices_wrap = shift_labels_wrap != -100
        valid_indices_base = shift_labels_base != -100
        valid_indices_correct = shift_labels_correct != -100
        batch_size, seq_length, hidden_size = shift_probs_base.size()
        device = shift_probs_base.device
        label_clamped_wrap = torch.clamp(shift_labels_wrap, min=0, max=hidden_size - 1)
        label_clamped_base = torch.clamp(shift_labels_base, min=0, max=hidden_size - 1)
        label_clamped_correct = torch.clamp(shift_labels_correct, min=0, max=hidden_size - 1)

        rows, cols = torch.meshgrid(torch.arange(batch_size, device=device), torch.arange(seq_length, device=device))

        probs_out_wrap = shift_probs_wrap[rows, cols, label_clamped_wrap]
        probs_out_base = shift_probs_base[rows, cols, label_clamped_base]
        probs_out_correct = shift_probs_correct[rows, cols, label_clamped_correct]

        valid_prob_wrap = probs_out_wrap[valid_indices_wrap].detach()
        valid_prob_base = probs_out_base[valid_indices_base]
        valid_prob_correct = probs_out_correct[valid_indices_correct].detach()

        ratio1 = (valid_prob_wrap / valid_prob_base)
        ratio2 = (valid_prob_wrap / valid_prob_correct)
        hidden_states_out_base = shift_hidden_states_base[valid_indices_base]

        unlike_indices = ((ratio1 > 1.5) & (ratio2 > 1.5) & (valid_prob_wrap > 0.01)).detach()
        unlike_amount = torch.sum(unlike_indices).item()


        all_indices = ratio1 >= 0
        all_amount = torch.sum(all_indices).item()
        clus_indices = self.token_cluster(hidden_states_out_base, unlike_indices).detach()

        threshold = torch.tensor(self.threshold)

        scale = (((1 / (1 + torch.exp(-(ratio1 - threshold))))) * 2).detach()
        scale2 = (valid_prob_correct ** 0.5).detach()
        if (unlike_amount / all_amount) > 0.06:
            scale2[all_indices] = 0

        valid_prob_base[unlike_indices] = 1 - valid_prob_base[unlike_indices]
        valid_prob_base[valid_prob_base == 0] += 1e-5  # avoid 0
        valid_lprob_base = torch.log(valid_prob_base)
        scale2[clus_indices] = 0
        scale2[unlike_indices] = scale[unlike_indices]

        valid_lprob_base[all_indices] = scale2 * valid_lprob_base[all_indices]

        loss_unlike = -torch.sum(valid_lprob_base) / valid_lprob_base.size(0)
        return loss_unlike

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


class NLFTPeftModelForCausalLM(PeftModel):
    """
    Peft model for causal language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.


    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModelForCausalLM, get_peft_config

        >>> config = {
        ...     "peft_type": "PREFIX_TUNING",
        ...     "task_type": "CAUSAL_LM",
        ...     "inference_mode": False,
        ...     "num_virtual_tokens": 20,
        ...     "token_dim": 1280,
        ...     "num_transformer_submodules": 1,
        ...     "num_attention_heads": 20,
        ...     "num_layers": 36,
        ...     "encoder_hidden_size": 1280,
        ...     "prefix_projection": False,
        ...     "postprocess_past_key_value_function": None,
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        >>> peft_model = PeftModelForCausalLM(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 1843200 || all params: 775873280 || trainable%: 0.23756456724479544
        ```
    """

    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            labels=None,
            input_ids_wrap=None,
            attention_mask_wrap=None,
            labels_wrap=None,
            input_ids_correct=None,
            attention_mask_correct=None,
            labels_correct=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            weight=None,
            polarity=None,
            **kwargs,
    ):
        peft_config = self.active_peft_config
        kwargs.update( {'weight': weight, 'polarity': polarity, "labels_wrap": labels_wrap, "labels_correct": labels_correct})
        input_ids = torch.cat([input_ids_wrap, input_ids, input_ids_correct], dim=0)
        attention_mask = torch.cat([attention_mask_wrap, attention_mask, attention_mask_correct], dim=0)
        if not peft_config.is_prompt_learning:
            if self.base_model.config.model_type == "mpt":
                if inputs_embeds is not None:
                    raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(
                input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values, **kwargs
            )
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def generate(self, **kwargs):
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if peft_config.is_prompt_learning:
            if model_kwargs.get("attention_mask", None) is not None:
                prefix_attention_mask = torch.ones(
                    model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
                ).to(model_kwargs["input_ids"].device)
                model_kwargs["attention_mask"] = torch.cat(
                    (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
                )

            if model_kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                model_kwargs["position_ids"] = None

            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
                past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                model_kwargs["past_key_values"] = past_key_values
            else:
                if model_kwargs["past_key_values"] is None:
                    inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                    prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                    prompts = prompts.to(inputs_embeds.dtype)
                    model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                    model_kwargs["input_ids"] = None

        return model_kwargs
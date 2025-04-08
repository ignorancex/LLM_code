# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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


from transformers import GPT2Config


class MemoryGPT2Config(GPT2Config):
    model_type = "memory_gpt2"

    def __init__(self, vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12, n_inner=None,
                 activation_function="gelu_new", resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
                 layer_norm_epsilon=1e-5, initializer_range=0.02, summary_type="cls_index", summary_use_proj=True,
                 summary_activation=None, summary_proj_to_labels=True, summary_first_dropout=0.1,
                 scale_attn_weights=True, use_cache=True, bos_token_id=50256, eos_token_id=50256,
                 scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False,
                 # custom arguments
                 add_memory_attention=False, n_memory_slots=64, add_memory_slot_selfattn=False, kmeans_memory=False,
                 deque_iters=10, window=1.0, freeze_memory=False,
                 **kwargs):

        self.add_memory_attention = add_memory_attention
        self.n_memory_slots = n_memory_slots
        self.add_memory_slots_selfattn = add_memory_slot_selfattn
        self.deque_iters = deque_iters
        self.kmeans_memory = kmeans_memory
        self.window = window
        self.freeze_memory = freeze_memory

        super().__init__(vocab_size, n_positions, n_embd, n_layer, n_head, n_inner, activation_function, resid_pdrop,
                         embd_pdrop, attn_pdrop, layer_norm_epsilon, initializer_range, summary_type, summary_use_proj,
                         summary_activation, summary_proj_to_labels, summary_first_dropout, scale_attn_weights,
                         use_cache, bos_token_id, eos_token_id, scale_attn_by_inverse_layer_idx,
                         reorder_and_upcast_attn, **kwargs)

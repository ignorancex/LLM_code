"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Transformer model definition."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from flax import linen as nn
from jax.sharding import Mesh
import numpy as np
import jax.numpy as jnp
# from jax.experimental.pallas.ops.tpu import flash_attention
from layers import attentions
from layers import embeddings
from layers import linears
from layers import normalizations
from layers import models
from layers import quantizations
from layers import initializers  # XD
from layers import mudd

import common_types
from typing import Optional

import max_logging

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn

AxisNames = common_types.AxisNames
BATCH = common_types.BATCH
KV_BATCH = common_types.KV_BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
KV_HEAD = common_types.KV_HEAD
D_KV = common_types.D_KV
KV_HEAD_DIM = common_types.KV_HEAD_DIM


Embed = embeddings.Embed
Attention = attentions.Attention
Quant = quantizations.AqtQuantization
nd_dense_init = initializers.nd_dense_init  # XD
DenseGeneral = linears.DenseGeneral  # XD
contant_dense_init = initializers.contant_dense_init  # lsp
NormalInitializer = initializers.nd_dense_init_normal # lsp
# -----------------------------------------
# The Decoder Layer specific for Llama2
# -----------------------------------------


class LlamaDecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""

  config: models.Config
  mesh: Mesh
  quant: Optional[Quant] = None

  def setup(self) -> None:  # XD
    cfg = self.config
    if not getattr(cfg, 'dense_conn', False): return
    norm_kwargs = {
                "dtype": cfg.dtype,
                "weight_dtype": cfg.weight_dtype,
                "name": "pre_dense_proj1_norm",
                "epsilon": cfg.normalization_layer_epsilon,
                }
    if not getattr(cfg, 'mudd_prenorm', False):
        max_logging.log(f'mudd_prenorm is False')
        norm_kwargs['scale_init'] = None # it means use scale is false

    self.pre_dense_proj1_norm = models.get_rmsnorm(**norm_kwargs)
    
    factor = 1
    i = int(self.name.split('_')[-1])  # name=f"layers_{i}"
    self.layer_inx = i
    C = 1 if cfg.dynamic_dense_fix_last_layer and i == cfg.num_decoder_layers-1 else len(cfg.dynamic_dense_type)
    dw_shape = (C, ((i + 1) * factor + 1))
    max_logging.log(f'dynamic_dense_hidden_expand-{i}: {cfg.dynamic_dense_hidden_expand[i]}')

    dynamic_dense_inter_dim = int(np.prod(dw_shape) * cfg.dynamic_dense_hidden_expand[i]) # lsp
    # if cfg.dynamic_dense_fix_last_layer and i== cfg.num_decoder_layers-1:
    #   dynamic_dense_inter_dim *= len(cfg.dynamic_dense_type)
    if cfg.dynamic_dense_hidden_round:  # default: round to 64 or 128
      # assert dynamic_dense_inter_dim < 128
      dynamic_dense_inter_dim = (dynamic_dense_inter_dim// 64 + 1) * 64

    kwargs = dict(
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      quant=self.quant,
    )
    self.dense_proj1 = DenseGeneral(
      dynamic_dense_inter_dim,
      kernel_init=nd_dense_init(1.0, "fan_in", "normal"),
      kernel_axes=('embed', 'kv'),
      use_bias=False,
      name='dynamic_dense_conn1',
      **kwargs
    )
    self.dense_activation = linears._convert_to_activation_function(cfg.dynamic_dense_act_cls)
    
    self.dense_proj2 = DenseGeneral(dw_shape, 
                                    kernel_init=contant_dense_init(0.0), 
                                    kernel_axes=('kv', None), 
                                    use_bias=False, 
                                    name='dynamic_dense_conn2', 
                                    **kwargs)
    if cfg.mudd_prenorm and cfg.mudd_postnorm:
      self.dense2_bias_init_value = 0.0
    else:
      self.dense2_bias_init_value = 1.0

    init_v = jnp.array([0] * ((i + 1) * factor) + [self.dense2_bias_init_value]).astype(cfg.weight_dtype) # dense_bias_init_method == 'current_only'
    init_v = init_v[None].repeat(C, 0)
    self.dense_proj2_bias = self.param(f"dense_proj2.bias", init_fn=lambda rng: init_v)

    if cfg.dynamic_mlp_dim:
      self.updated_mlp_dim = round(cfg.mlp_dim * (i / (cfg.num_decoder_layers - 1) + 0.5) / 128) * 128 
    else:
      self.updated_mlp_dim = cfg.mlp_dim
    max_logging.log(f'updated_mlp_dim: {self.updated_mlp_dim}')

  @nn.compact
  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      hids=None,
  ):
    cfg = self.config
    mesh = self.mesh

    if self.config.mudd_in_layer:
        if self.layer_inx == 0: # first layer
            inputs = [inputs] * len(self.config.dynamic_dense_type)
        else:
            inputs, hids = mudd.Compose(self.config, self.mesh, self.quant, self.layer_inx-1, name=f'compose_{self.layer_inx-1}')(inputs, hids) # lsp
    
    def norm(inputs, name_suffix=''):  # XD
        name = 'pre_self_attention_layer_norm' + name_suffix
        lnx = models.get_rmsnorm(name=name, cfg=cfg)(inputs)
        return nn.with_logical_constraint(lnx, ("activation_batch", "activation_length", "activation_embed"))
    max_logging.log(f'dynamic_dense_type: {cfg.dynamic_dense_type}')
    if cfg.dynamic_dense_type == 'qkvm': # XD
      assert isinstance(inputs, (tuple, list, Array)) and len(inputs) == 4 # lsp: Array also support, but C dimenson must be in 0.
      inputs = [nn.with_logical_constraint(i, ("activation_batch", "activation_length", "activation_embed")) for i in inputs]
      lnx_q, *lnx_kv = [norm(inp, f'_{name_suffix}') for inp, name_suffix in zip(inputs[:3], 'qkv')]
      inputs = inputs[-1] # m: bld
    else:
      inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))
      lnx_q = lnx_kv = norm(inputs)

    # Self-attention block
    attention_layer = Attention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        mesh=mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        name="self_attention",
        quant=self.quant,
        kernel_init=NormalInitializer(0.006), # lsp
        float32_qk_product = False,  # computes logits in float32 for stability.
        float32_logits = True,
        quantize_kvcache=cfg.quantize_kvcache,
        prefill_cache_axis_order=tuple([int(i) for i in cfg.prefill_cache_axis_order.split(",")]),
        ar_cache_axis_order=tuple([int(i) for i in cfg.ar_cache_axis_order.split(",")]),
        compute_axis_order=tuple([int(i) for i in cfg.compute_axis_order.split(",")]),
        reshape_q=cfg.reshape_q,
        kv_quant_axis=cfg.kv_quant_axis,
    )

    attention_lnx = attention_layer(
        lnx_q, # XD lnx,
        lnx_kv, # XD lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )

    attention_lnx = nn.with_logical_constraint(attention_lnx, ("activation_batch", "activation_length", "activation_embed"))
    intermediate_inputs = inputs + attention_lnx

    # Fully Connected
    hidden_states = models.get_rmsnorm(name="post_self_attention_layer_norm", cfg=cfg)(intermediate_inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, ("activation_batch", "activation_length", "activation_embed"))


    # MLP block.
    mlp_lnx = linears.MlpBlock(
        intermediate_dim=self.updated_mlp_dim if cfg.dynamic_mlp_dim else cfg.mlp_dim,  # lsp
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp",
        config=cfg,
        quant=self.quant,
        kernel_init=NormalInitializer(0.006),  # lsp
    )(hidden_states, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_length", "activation_embed"))

    layer_output = mlp_lnx + intermediate_inputs

    layer_output = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(layer_output, deterministic=deterministic)

    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_length", "activation_embed"),
    )

    # if cfg.record_internal_nn_metrics:
    #   self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
    #   self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
    #   self.sow(
    #       "intermediates",
    #       "activation_fraction_zero",
    #       jnp.sum(layer_output == 0) / jnp.size(layer_output),
    #   )
      
    if self.config.mudd_in_layer:
        if self.layer_inx == self.config.base_num_decoder_layers-1: # last layer
            layer_output, hids = mudd.Compose(self.config, self.mesh, self.quant, self.layer_inx, name=f'compose_{self.layer_inx}')(layer_output, hids) # lsp
        return layer_output, hids
    else:
        if cfg.dynamic_dense_type == 'qkvm': # XD lsp
            x_out_normed = self.pre_dense_proj1_norm(layer_output)
            dense_w_inner = self.dense_activation(self.dense_proj1(x_out_normed))
            dyn_dense_kernel_out = self.dense_proj2(dense_w_inner)
            dyn_dense_w = dyn_dense_kernel_out + self.dense_proj2_bias.astype(dyn_dense_kernel_out.dtype) # dense_proj2_bias初始化出来时fp32
     

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output if cfg.dynamic_dense_type != 'qkvm' else (layer_output, dyn_dense_w)  # XD , lsp
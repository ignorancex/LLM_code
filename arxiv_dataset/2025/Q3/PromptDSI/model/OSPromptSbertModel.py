import math
from typing import Optional

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from transformers import MPNetPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.mpnet.modeling_mpnet import (
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    MPNET_INPUTS_DOCSTRING,
    MPNetEmbeddings,
    MPNetIntermediate,
    MPNetOutput,
    MPNetPooler,
)
from transformers.pytorch_utils import (
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)

logger = logging.get_logger(__name__)

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    # token_embeddings = model_output[
    #     0
    # ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class PromptMPNetAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = PromptMPNetSelfAttention(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.attn.num_attention_heads,
            self.attn.attention_head_size,
            self.pruned_heads,
        )

        self.attn.q = prune_linear_layer(self.attn.q, index)
        self.attn.k = prune_linear_layer(self.attn.k, index)
        self.attn.v = prune_linear_layer(self.attn.v, index)
        self.attn.o = prune_linear_layer(self.attn.o, index, dim=1)

        self.attn.num_attention_heads = self.attn.num_attention_heads - len(heads)
        self.attn.all_head_size = (
            self.attn.attention_head_size * self.attn.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        position_bias=None,
        output_attentions=False,
        prompt=None,
        **kwargs,
    ):
        self_outputs = self.attn(
            hidden_states,
            attention_mask,
            head_mask,
            position_bias,
            output_attentions=output_attentions,
            prompt=prompt,
        )
        attention_output = self.LayerNorm(self.dropout(self_outputs[0]) + hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class PromptMPNetSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.q = nn.Linear(config.hidden_size, self.all_head_size)
        self.k = nn.Linear(config.hidden_size, self.all_head_size)
        self.v = nn.Linear(config.hidden_size, self.all_head_size)
        self.o = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        position_bias=None,
        output_attentions=False,
        prompt=None,
        **kwargs,
    ):
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # Adding prefix prompt
        B, N, C = hidden_states.shape

        if prompt is not None:
            # prefix key, value
            prompt = prompt.permute(
                1, 0, 3, 2, 4
            ).contiguous()  # 2, B, num_heads, prompt_length, C // num_heads
            key_prefix = prompt[
                0
            ]  # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[
                1
            ]  # B, num_heads, prompt_length, embed_dim // num_heads

            expected_shape = (
                B,
                self.num_attention_heads,
                C // self.num_attention_heads,
            )

            assert (
                key_prefix.shape[0],
                key_prefix.shape[1],
                key_prefix.shape[3],
            ) == expected_shape, f"key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}"
            assert (
                value_prefix.shape[0],
                value_prefix.shape[1],
                value_prefix.shape[3],
            ) == expected_shape, f"value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}"
            # k = torch.cat([key_prefix, k], dim=2)
            # v = torch.cat([value_prefix, v], dim=2)
            # Switch to appending instead of prepending (Test)
            k = torch.cat([k, key_prefix], dim=2)
            v = torch.cat([v, value_prefix], dim=2)
        # Done adding prefix prompt

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply relative position embedding (precomputed in PromptMPNetEncoder) if provided.
        if position_bias is not None:
            attention_scores += position_bias

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        c = torch.matmul(attention_probs, v)

        c = c.permute(0, 2, 1, 3).contiguous()
        new_c_shape = c.size()[:-2] + (self.all_head_size,)
        c = c.view(*new_c_shape)

        o = self.o(c)

        outputs = (o, attention_probs) if output_attentions else (o,)
        return outputs


class PromptMPNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prefix = True
        self.config = config
        self.attention = PromptMPNetAttention(config)
        self.intermediate = MPNetIntermediate(config)
        self.output = MPNetOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        position_bias=None,
        output_attentions=False,
        prompt=None,
        **kwargs,
    ):
        if self.prefix and prompt is not None:
            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                head_mask,
                position_bias=position_bias,
                output_attentions=output_attentions,
                prompt=prompt,
            )
        else:
            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                head_mask,
                position_bias=position_bias,
                output_attentions=output_attentions,
            )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class PromptMPNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_heads = config.num_attention_heads
        self.layer = nn.ModuleList(
            [PromptMPNetLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.relative_attention_bias = nn.Embedding(
            config.relative_attention_num_buckets, self.n_heads
        )
        self.prompt_feature_layer = config.prompt_feature_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        original_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        e_prompt_layer_idx=None,
        e_prompt=None,
        f=0,
        previous_task_key_centroids=None,
        task_id=None,
        **kwargs,
    ):
        position_bias = self.compute_position_bias(hidden_states)
        e_prompt_position_bias = self.compute_position_bias_eprompt(
            hidden_states, attention_mask[0].shape[-1]
        )
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        e_prompt_counter = -1
        e_prompt_attention_mask, normal_attention_mask = attention_mask
        cls_features = None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i == self.prompt_feature_layer:
                cls_features = mean_pooling(
                    all_hidden_states[self.prompt_feature_layer],
                    original_attention_mask,
                )
                cls_features = F.normalize(cls_features, p=2, dim=1)
                res = e_prompt(
                    f=f,
                    task_id=task_id,
                    cls_features=cls_features,
                    train=self.training,
                    previous_task_key_centroids=previous_task_key_centroids,
                )
                prompts = res["batched_prompt"]

            if i in e_prompt_layer_idx:
                e_prompt_counter += 1
                # cls_features = mean_pooling(
                #     all_hidden_states[self.prompt_feature_layer],
                #     original_attention_mask,
                # )
                # cls_features = F.normalize(cls_features, p=2, dim=1)
                # res = e_prompt(
                #     f=f,
                #     task_id=task_id,
                #     cls_features=cls_features,
                #     train=self.training,
                #     previous_task_key_centroids=previous_task_key_centroids,
                # )
                # prompts = res["batched_prompt"]
                layer_outputs = layer_module(
                    hidden_states,
                    e_prompt_attention_mask,
                    head_mask[i],
                    e_prompt_position_bias,
                    output_attentions=output_attentions,
                    prompt=prompts[e_prompt_counter],
                    **kwargs,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    normal_attention_mask,
                    head_mask[i],
                    position_bias,
                    output_attentions=output_attentions,
                    **kwargs,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return (
                tuple(
                v
                for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None
            ),
            res,
        )
        return (
            BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            ),
            res,
        )

    def compute_position_bias(self, x, position_ids=None, num_buckets=32):
        bsz, qlen, klen = x.size(0), x.size(1), x.size(1)
        if position_ids is not None:
            context_position = position_ids[:, :, None]
            memory_position = position_ids[:, None, :]
        else:
            context_position = torch.arange(qlen, dtype=torch.long)[:, None]
            memory_position = torch.arange(klen, dtype=torch.long)[None, :]

        relative_position = memory_position - context_position

        rp_bucket = self.relative_position_bucket(
            relative_position, num_buckets=num_buckets
        )
        rp_bucket = rp_bucket.to(x.device)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        values = values.expand((bsz, -1, qlen, klen)).contiguous()
        return values

    def compute_position_bias_eprompt(
        self, x, eprompt_len, position_ids=None, num_buckets=32
    ):
        bsz, qlen, klen = x.size(0), x.size(1), eprompt_len
        if position_ids is not None:
            context_position = position_ids[:, :, None]
            memory_position = position_ids[:, None, :]
        else:
            context_position = torch.arange(qlen, dtype=torch.long)[:, None]
            memory_position = torch.arange(klen, dtype=torch.long)[None, :]

        relative_position = memory_position - context_position

        rp_bucket = self.relative_position_bucket(
            relative_position, num_buckets=num_buckets
        )
        rp_bucket = rp_bucket.to(x.device)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        values = values.expand((bsz, -1, qlen, klen)).contiguous()
        return values

    @staticmethod
    def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)

        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )
        ret += torch.where(is_small, n, val_if_large)
        return ret


class MPNetModel(MPNetPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = MPNetEmbeddings(config)
        self.encoder = PromptMPNetEncoder(config)
        self.pooler = MPNetPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(
        MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ): #-> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class OSPromptSbertModel(MPNetModel):
    def __init__(
        self,
        config,
        eprompt_class,
    ):
        super().__init__(config)
        self.config = config
        self.prompt_pool = config.prompt_pool
        self.head_type = config.head_type
        self.use_prompt_mask = config.use_prompt_mask

        # e_prompt
        self.use_e_prompt = True
        self.prefix_e_prompt = True
        self.e_prompt_layer_idx = config.e_prompt_layer_idx
        num_e_prompt = (
            len(self.e_prompt_layer_idx) if self.e_prompt_layer_idx is not None else 0
        )
        assert (
            config.e_prompt_layer_idx[0] >= config.prompt_feature_layer
        ), "The e-prompt layer should be after the feature layer"

        self.prompt = eprompt_class(
            length=config.prompt_length,
            num_layers=num_e_prompt,
            **vars(config),
        )
        self.total_prompt_len = 0
        if self.prompt_pool:
            if not self.prefix_e_prompt:
                self.total_prompt_len += (
                    config.prompt_length * config.top_k * len(self.e_prompt_layer_idx)
                )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ### Adding prompt params ###
        task_id=None,
        train=False,
        f=None,
        previous_task_key_centroids=None,
        **kwargs,
    ):  #-> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        original_attention_mask = attention_mask

        e_prompt_prefix_attention_mask = (
            torch.ones(
                batch_size,
                self.config.prompt_length * self.prompt.top_k,
                device=attention_mask.device,
            )
            if self.use_e_prompt
            else None
        )
        e_prompt_extended_attention_mask: torch.Tensor = (
            (
                self.get_extended_attention_mask(
                    torch.cat((e_prompt_prefix_attention_mask, attention_mask), dim=1),
                    input_shape,
                )
            )
            if self.use_e_prompt
            else None
        )    

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        extended_attention_mask = (
            e_prompt_extended_attention_mask,
            extended_attention_mask,
        )

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs, res = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            original_attention_mask=original_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            e_prompt_layer_idx=self.e_prompt_layer_idx,
            e_prompt=self.prompt,
            f=f,
            previous_task_key_centroids=previous_task_key_centroids,
            task_id=task_id,
        )
        sequence_output = encoder_outputs[0]
        res["x"] = sequence_output
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:], res

        return (
            BaseModelOutputWithPooling(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            ),
            res,
        )

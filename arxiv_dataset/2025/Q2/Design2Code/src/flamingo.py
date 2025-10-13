import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch_geometric.nn import GCNConv, aggr
from src.helpers import PerceiverResampler
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from src.utils import apply_with_stopping_condition


class Flamingo(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        graph_token_id: int,
        vis_dim: int,
        graph_dim: int,
        enable_graph_input: bool=False,
        cross_attn_every_n_layers: int = 3,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <image>
            graph_token_id (int): Token id for <graph>
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
        """
        super().__init__()
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.graph_token_id = graph_token_id
        self.vis_dim = vis_dim
        self.graph_dim = graph_dim
        if hasattr(lang_encoder.config, "d_model"):
            self.lang_dim = lang_encoder.config.d_model  # mpt uses d_model
        else:
            self.lang_dim = lang_encoder.config.hidden_size

        self.vision_encoder = vision_encoder.visual
        self.perceiver = PerceiverResampler(dim=self.vis_dim)
        self.enable_graph_input = enable_graph_input
        if enable_graph_input:
            self.graph_encoder = GCN(graph_dim)
        self.lang_encoder = lang_encoder
        
        self.lang_encoder.init_flamingo(
            media_token_id=media_token_id,
            graph_token_id=graph_token_id,
            lang_hidden_size=self.lang_dim,
            vis_hidden_size=self.vis_dim,
            dim_graph=graph_dim,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            gradient_checkpointing=gradient_checkpointing,
            enable_graph_input=self.enable_graph_input,
        )
        self._use_gradient_checkpointing = gradient_checkpointing
        self.perceiver._use_gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        graph_x: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        clear_media_conditioned_layers: bool = True,
        clear_graph_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_media_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        assert (
            self.lang_encoder.initialized_flamingo
        ), "Flamingo layers are not initialized. Please call `init_flamingo` first."

        assert (
            self.lang_encoder._use_cached_vision_x or vision_x is not None
        ), "Must provide either vision_x or have precached media using cache_media()."

        if self.enable_graph_input:
            assert (
                self.lang_encoder._use_cached_graph_x or graph_x is not None
            ), "Must provide either graph_x or have precached graph using cache_graph()."

        if self.lang_encoder._use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when media has been cached using cache_media(). Try uncache_media() first."
            assert self.lang_encoder.is_media_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_vision_x(vision_x=vision_x)
            self._condition_media_locations(input_ids=lang_x)

        if self.enable_graph_input:
            if self.lang_encoder._use_cached_graph_x:
                assert (
                    graph_x is None
                ), "Expect graph_x to be None when graph has been cached using cache_graph(). Try uncache_graph() first."
                assert self.lang_encoder.is_graph_conditioned()
            else:
                self._encode_graph_x(graph_x=graph_x)
                self._condition_graph_locations(input_ids=lang_x)


        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if clear_media_conditioned_layers:
            self.lang_encoder.clear_media_conditioned_layers()

        if self.enable_graph_input and clear_graph_conditioned_layers:
            self.lang_encoder.clear_graph_conditioned_layers()

        return output

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        graph_x = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                images in the same chunk are collated along T_img, and frames are collated along F
                currently only F=1 is supported (single-frame videos)
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            **kwargs: see generate documentation in Hugging Face CausalLM models. Some notable kwargs:
                max_length (int, optional): Maximum length of the output. Defaults to None.
                attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
                num_beams (int, optional): Number of beams. Defaults to 1.
                max_new_tokens (int, optional): Maximum new tokens. Defaults to None.
                temperature (float, optional): Temperature. Defaults to 1.0.
                top_k (int, optional): Top k. Defaults to 50.
                top_p (float, optional): Top p. Defaults to 1.0.
                no_repeat_ngram_size (int, optional): No repeat ngram size. Defaults to 0.
                length_penalty (float, optional): Length penalty. Defaults to 1.0.
                num_return_sequences (int, optional): Number of return sequences. Defaults to 1.
                do_sample (bool, optional): Do sample. Defaults to False.
                early_stopping (bool, optional): Early stopping. Defaults to False.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        num_beams = kwargs.pop("num_beams", 1)
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)

        self.lang_encoder._use_cached_vision_x = True
        self._encode_vision_x(vision_x=vision_x)
        
        if graph_x != None:
            self.lang_encoder._use_cached_graph_x = True
            self._encode_graph_x(graph_x=graph_x)

        eos_token_id = kwargs.pop("eos_token_id", self.eoc_token_id)
        output = self.lang_encoder.generate(
            input_ids=lang_x,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            num_beams=num_beams,
            **kwargs,
        )

        self.lang_encoder.clear_media_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False
        if graph_x != None:
            self.lang_encoder.clear_graph_conditioned_layers()
            self.lang_encoder._use_cached_graph_x = False
        return output

    def _encode_graph_x(self, graph_x: torch.Tensor):
        graph_x = self.graph_encoder(graph_x)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_graph_x(graph_x)
            
    
    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_encoder(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        vision_x = self.perceiver(vision_x)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)


    def wrap_fsdp(self, wrapper_kwargs, device_id):
        """
        Manually wraps submodules for FSDP and move other parameters to device_id.

        Why manually wrap?
        - all parameters within the FSDP wrapper must have the same requires_grad.
            We have a mix of frozen and unfrozen parameters.
        - model.vision_encoder.visual needs to be individually wrapped or encode_vision_x errors
            See: https://github.com/pytorch/pytorch/issues/82461#issuecomment-1269136344

        The rough wrapping structure is:
        - FlamingoModel
            - FSDP(FSDP(vision_encoder))
            - FSDP(FSDP(perceiver))
            - lang_encoder
                - FSDP(FSDP(input_embeddings))
                - FlamingoLayers
                    - FSDP(FSDP(gated_cross_attn_layer))
                    - FSDP(FSDP(decoder_layer))
                - FSDP(FSDP(output_embeddings))
                - other parameters

        Known issues:
        - Our FSDP strategy is not compatible with tied embeddings. If the LM embeddings are tied,
            train with DDP or set the --freeze_lm_embeddings flag to true.
        - With FSDP + gradient ckpting, one can increase the batch size with seemingly no upper bound.
            Although the training curves look okay, we found that downstream performance dramatically
            degrades if the batch size is unreasonably large (e.g., 100 MMC4 batch size for OPT-125M).

        FAQs about our FSDP wrapping strategy:
        Why double wrap?
        As of torch==2.0.1, FSDP's _post_forward_hook and _post_backward_hook
        only free gathered parameters if the module is NOT FSDP root.

        Why unfreeze the decoder_layers?
        See https://github.com/pytorch/pytorch/issues/95805
        As of torch==2.0.1, FSDP's _post_backward_hook is only registed if the flat param
        requires_grad=True. We need the postback to fire to avoid OOM.
        To effectively freeze the decoder layers, we exclude them from the optimizer.

        What is assumed to be frozen v. unfrozen?
        We assume that the model is being trained under normal Flamingo settings
        with these lines being called in factory.py:
            ```
            # Freeze all parameters
            model.requires_grad_(False)
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

            # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
            model.perceiver.requires_grad_(True)
            model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
            [optional] model.lang_encoder.get_input_embeddings().requires_grad_(True)
            ```
        """
        # unfreeze the decoder layers
        for block in self.lang_encoder.old_decoder_blocks:
            block.requires_grad_(True)

        # wrap in FSDP
        with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):
            self.perceiver = wrap(wrap(self.perceiver))
            self.lang_encoder.old_decoder_blocks = nn.ModuleList(
                wrap(wrap(block)) for block in self.lang_encoder.old_decoder_blocks
            )
            self.lang_encoder.gated_cross_attn_layers = nn.ModuleList(
                wrap(wrap(layer)) if layer is not None else None
                for layer in self.lang_encoder.gated_cross_attn_layers
            )
            self.lang_encoder.init_flamingo_layers(self._use_gradient_checkpointing)
            self.lang_encoder.set_input_embeddings(
                wrap(wrap(self.lang_encoder.get_input_embeddings()))
            )
            self.lang_encoder.set_output_embeddings(
                wrap(wrap(self.lang_encoder.get_output_embeddings()))
            )
            self.vision_encoder = wrap(wrap(self.vision_encoder))  # frozen

        # manually move non-FSDP managed parameters to device_id
        # these are all in lang_encoder
        apply_with_stopping_condition(
            module=self.lang_encoder,
            apply_fn=lambda m: m.to(device_id),
            apply_condition=lambda m: len(list(m.children())) == 0,
            stopping_condition=lambda m: isinstance(m, FSDP),
        )

        # exclude the original decoder layers from the optimizer
        for block in self.lang_encoder.old_decoder_blocks:
            for p in block.parameters():
                p.exclude_from_optimizer = True

        # set up clip_grad_norm_ function
        def clip_grad_norm_(max_norm):
            self.perceiver.clip_grad_norm_(max_norm)
            for layer in self.lang_encoder.gated_cross_attn_layers:
                if layer is not None:
                    layer.clip_grad_norm_(max_norm)
            self.lang_encoder.get_input_embeddings().clip_grad_norm_(max_norm)

        self.clip_grad_norm_ = clip_grad_norm_

    def _condition_graph_locations(self, input_ids: torch.Tensor):
        """
        Compute the graph token locations from lang_x and condition the language model on these.
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
        """
        graph_locations = input_ids == self.graph_token_id

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_graph_locations(graph_locations)        
    
    def _condition_media_locations(self, input_ids: torch.Tensor):
        """
        Compute the media token locations from lang_x and condition the language model on these.
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
        """
        media_locations = input_ids == self.media_token_id

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_media_locations(media_locations)
            
            
    def _condition_graph_locations(self, input_ids: torch.Tensor):
        """
        Compute the graph token locations from lang_x and condition the language model on these.
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
        """
        graph_locations = input_ids == self.graph_token_id

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_graph_locations(graph_locations)


    def cache_media(self, input_ids: torch.Tensor, vision_x: torch.Tensor):
        """
        Pre-cache a prompt/sequence of images / text for log-likelihood evaluations.
        All subsequent calls to forward() will generate attending to the LAST
        image in vision_x.
        This is not meant to be used to cache things for generate().
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)
        """
        self._encode_vision_x(vision_x=vision_x)
        self._condition_media_locations(input_ids=input_ids)
        self.lang_encoder._use_cached_vision_x = True

    def uncache_media(self):
        """
        Clear all conditioning.
        """
        self.lang_encoder.clear_media_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False


"""PREPARE GNN MODEL
"""
class GCN(nn.Module):
    def __init__(self, num_features, num_layers=8):
        """
        Initializes the GCN model with the specified number of features and layers.

        Args:
            num_features (int): The number of input features.
            num_layers (int): The number of GCN layers (default is 8).

        Returns:
            None
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                GCNConv(num_features, 128),
                GCNConv(128, num_features)
            ]))
        self.global_pool = aggr.MeanAggregation()

    def forward(self, pyg_graph):
        x, edge_index = pyg_graph.x, pyg_graph.edge_index
        for conv1, conv2 in self.layers:
            x = conv1(x, edge_index)
            x = F.relu(x)
            x = conv2(x, edge_index)
        out = self.global_pool(x, pyg_graph.batch)
        return out
    


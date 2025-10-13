import torch.nn as nn
from src.helpers import GatedCrossAttentionBlock
from src.utils import getattr_recursive, setattr_recursive


class FlamingoLayer(nn.Module):
    """
    FlamingoLayer is a wrapper around the GatedCrossAttentionBlock and DecoderLayer.
    """

    def __init__(
        self, gated_cross_attn_layer, decoder_layer, gradient_checkpointing=False, enable_graph_input=None,
    ):
        super().__init__()
        self.enable_graph_input = enable_graph_input
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None
        if self.enable_graph_input:
            self.graph_x = None
            self.graph_locations = None
        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer._use_gradient_checkpointing = (
                gradient_checkpointing
            )
        self.decoder_layer._use_gradient_checkpointing = gradient_checkpointing

    def is_media_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None and self.media_locations is not None
    
    def is_graph_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""   
        return self.graph_x is not None and self.graph_locations is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x):
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations):
        self.media_locations = media_locations

    def condition_use_cached_media(self, use_cached_media):
        self.use_cached_media = use_cached_media

    def condition_graph_x(self, graph_x):
        self.graph_x = graph_x
        
    def condition_graph_locations(self, graph_locations):
        self.graph_locations = graph_locations
        
    def condition_use_cached_graph(self, use_cached_graph):
        self.use_cached_graph = use_cached_graph

    def forward(
        self,
        lang_x,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        # Cross attention
        if self.gated_cross_attn_layer is not None:
            if self.vis_x is None:
                raise ValueError("vis_x must be conditioned before forward pass")
            if self.media_locations is None:
                raise ValueError("media_locations must be conditioned before forward pass")
            if self.enable_graph_input and self.graph_x is None:
                raise ValueError("graph_x must be conditioned before forward pass")
            if self.enable_graph_input and self.graph_locations is None:
                raise ValueError("graph_locations must be conditioned before forward pass")

            if self.enable_graph_input:
                lang_x = self.gated_cross_attn_layer(
                    lang_x,
                    self.vis_x,
                    self.graph_x,
                    media_locations=self.media_locations,
                    graph_locations=self.graph_locations,
                    use_cached_media=self.use_cached_media,
                )
            else:
                lang_x = self.gated_cross_attn_layer(
                    lang_x,
                    self.vis_x,
                    media_locations=self.media_locations,
                    use_cached_media=self.use_cached_media,
                )

        # Normal decoder layer
        lang_x = self.decoder_layer(
            lang_x, attention_mask=attention_mask, **decoder_layer_kwargs
        )
        return lang_x


class FlamingoLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_flamingo(
        self,
        media_token_id,
        lang_hidden_size,
        vis_hidden_size,
        cross_attn_every_n_layers,
        gradient_checkpointing,
        graph_token_id=None,
        dim_graph=None,
        enable_graph_input=None,
    ):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """
        self.old_decoder_blocks = self._get_decoder_layers()
        self.enable_graph_input = enable_graph_input
        if enable_graph_input:
            self.gated_cross_attn_layers = nn.ModuleList(
                [
                    GatedCrossAttentionBlock(
                        dim=lang_hidden_size, dim_visual=vis_hidden_size, dim_graph=dim_graph, enable_graph_input=enable_graph_input
                    )
                    if (layer_idx + 1) % cross_attn_every_n_layers == 0
                    else None
                    for layer_idx, _ in enumerate(self._get_decoder_layers())
                ]
            )
            self.graph_token_id = graph_token_id
            self._use_cached_graph_x = False
        else:
            self.gated_cross_attn_layers = nn.ModuleList(
                [
                    GatedCrossAttentionBlock(
                        dim=lang_hidden_size, dim_visual=vis_hidden_size, enable_graph_input=enable_graph_input
                    )
                    if (layer_idx + 1) % cross_attn_every_n_layers == 0
                    else None
                    for layer_idx, _ in enumerate(self._get_decoder_layers())
                ]
            )

        self.init_flamingo_layers(gradient_checkpointing)
        self.media_token_id = media_token_id
        self.initialized_flamingo = True
        self._use_cached_vision_x = False

    def init_flamingo_layers(self, gradient_checkpointing):
        """
        Re initializes the FlamingoLayers.
        Propagates any changes made to self.gated_corss_attn_layers or self.old_decoder_blocks
        """
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    FlamingoLayer(
                        gated_cross_attn_layer, decoder_layer, gradient_checkpointing, self.enable_graph_input
                    )
                    for gated_cross_attn_layer, decoder_layer in zip(
                        self.gated_cross_attn_layers, self.old_decoder_blocks
                    )
                ]
            )
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        """Condition the Flamingo layers on the media locations before forward()"""
        if not self.initialized_flamingo:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo` first."
            )

        media_locations = input_ids == self.media_token_id
        if self.enable_graph_input:
            graph_locations = input_ids == self.graph_token_id
            use_cached_graph_locations = (
                self._use_cached_graph_x
                and self.is_graph_conditioned()
                and not graph_locations.any()
            )
            for layer in self._get_decoder_layers():
                if not use_cached_graph_locations:
                    layer.condition_graph_locations(graph_locations)
                layer.condition_use_cached_graph(use_cached_graph_locations)

        # if there are media already cached and we're generating and there are no media tokens in the input,
        # we'll assume that ALL input tokens should attend to the last previous media that is cached.
        # this is especially important for HF generate() compatibility, since generate() calls forward()
        # repeatedly one token at a time (with no media tokens).
        # without this check, the model would not attend to any images when generating (after the first token)
        use_cached_media_locations = (
            self._use_cached_vision_x
            and self.is_media_conditioned()
            and not media_locations.any()
        )

        for layer in self._get_decoder_layers():
            if not use_cached_media_locations:
                layer.condition_media_locations(media_locations)
            layer.condition_use_cached_media(use_cached_media_locations)

        # package arguments for the other parent's forward. since we don't know the order of the arguments,
        # make them all kwargs
        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        return super().forward(**kwargs)  # Call the other parent's forward method

    def is_media_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_media_conditioned() for l in self._get_decoder_layers())

    def is_graph_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_graph_conditioned() for l in self._get_decoder_layers())

    def clear_media_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
            layer.condition_use_cached_media(None)
            
    def clear_graph_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_graph_x(None)
            layer.condition_graph_locations(None)
            layer.condition_use_cached_graph(None)

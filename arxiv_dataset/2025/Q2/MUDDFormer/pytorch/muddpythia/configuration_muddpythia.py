from transformers.configuration_utils import PretrainedConfig
from typing import Optional


class MUDDPythiaConfig(PretrainedConfig):
    model_type = "muddpythia"

    '''
    MUDDPythiaConfig is a config class for MUDDPythia, which is adpated from https://github.com/pytorch-labs/gpt-fast/blob/main/model.py#L21
    '''
    def __init__(
        self,
        block_size: int = 2048,
        vocab_size: int = 32000,
        n_layer: int = 32,
        n_head: int = 32,
        dim: int = 2560,
        intermediate_size: int = None,
        n_local_heads: int = -1,
        head_dim: int = 64,
        rope_base: float = 10000,
        norm_eps: float = 1e-5,
        use_gradient_checkpointing: bool = False,
        is_training: bool = False,
        use_qk_norm: bool = False ,
        pad_token_id: Optional[int]= None,
        use_parallel_residual: bool =True,
        use_linear_bias: bool = True,
        rotary_pct: float = 0.25,
        bos_token_id: int =1,
        eos_token_id: int =2,
        tie_word_embeddings: bool =False,
        use_layer_cache: bool = True,
        dense: bool = True,
        dynamic_dense: bool = True,
        sepln: bool = True,
        dense_type: str = 'qkvr',
        expand_last: bool = False,
        round64: bool = False,
        **kwargs,
    ):
        self.block_size=block_size
        self.vocab_size=vocab_size
        self.n_layer=n_layer
        self.n_head=n_head
        self.dim=dim
        self.intermediate_size=intermediate_size
        self.n_local_heads=n_local_heads
        self.head_dim=head_dim
        self.rope_base=rope_base
        self.norm_eps=norm_eps
        self.use_gradient_checkpointing=use_gradient_checkpointing
        self.is_training=is_training
        self.use_qk_norm=use_qk_norm
        self.use_parallel_residual = use_parallel_residual
        self.use_linear_bias = use_linear_bias
        self.rotary_pct = rotary_pct

        self.use_layer_cache= use_layer_cache
        self.dense= dense
        self.dynamic_dense= dynamic_dense
        self.sepln= sepln
        self.dense_type=dense_type
        self.expand_last= expand_last
        self.round64 = round64
        # post init
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            self.intermediate_size =  4 * self.dim 
        self.head_dim = self.dim // self.n_head

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

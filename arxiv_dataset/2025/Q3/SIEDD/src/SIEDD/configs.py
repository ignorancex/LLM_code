from pydantic import BaseModel, Field, model_validator
from typing import Literal, Any
from fractions import Fraction
from pathlib import Path


class SineConfig(BaseModel):
    w0: float = 30.0
    c: float = 6.0
    finer: bool = False
    linear: bool = False
    power: float | None = None


LossType = Literal["mse", "l1"]
# muon optimizer doesn't work
OptimizerType = Literal["adam", "schedule_free", "grokadamw", "adam8bit", "muon"]
ActivationType = (
    Literal[
        "none",
        "linear",
        "relu",
        "leakyrelu",
        "tanh",
        "sigmoid",
        "selu",
        "elu",
        "softplus",
        "gelu",
    ]
    | SineConfig
)
CodingType = Literal["huffman", "ans", "none"]  # ans does not work
DistType = Literal["gaussian", "laplacian", "minmax"]  # unused
LRSchedulerType = Literal[
    "LambdaLR",
    "MultiplicativeLR",
    "StepLR",
    "MultiStepLR",
    "ConstantLR",
    "LinearLR",
    "ExponentialLR",
    "SequentialLR",
    "CosineAnnealingLR",
    "ChainedScheduler",
    "ReduceLROnPlateau",
    "CyclicLR",
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "PolynomialLR",
    "LRScheduler",
]


class LMCConfig(BaseModel):
    # Unused
    a: float = 2e1
    b: float = 2e-2
    minpct: float = 0.1
    lossminpc: float = 0.1
    alpha: float = 0.7


SamplingType = Literal["uniform", "loss", "unique", "edge"] | LMCConfig | None


class NoPosEncode(BaseModel):
    name: Literal["NoPosEncode"] = "NoPosEncode"
    dim_in: int = 2
    dim_out: int = 2


class NeRFConfig(BaseModel):
    name: Literal["nerf"] = "nerf"
    dim_in: int = 2
    dim_out: int = 16
    include_coord: bool = True
    trainable: bool = False

    @property
    def param_size(self):
        if self.include_coord:
            return (self.dim_out - self.dim_in) // (2 * self.dim_in)
        return self.dim_out // (2 * self.dim_in)

    @model_validator(mode="after")
    def check_shapes(self):
        if self.include_coord:
            if (self.dim_out - self.dim_in) % (2 * self.dim_in) != 0:
                raise ValueError("Invaild dim_out for include_coord=True")
        elif not self.include_coord and self.dim_out % (2 * self.dim_in) != 0:
            raise ValueError(f"dim_out must be divisible by {2 * self.dim_in}")
        return self


class FourierConfig(BaseModel):
    # To differentiate fourier and gaussian with pydantic
    name: Literal["fourier"] = "fourier"
    dim_in: int = 2
    dim_out: int = 256
    pos_scale: float = 10.0
    trainable: bool = False

    @model_validator(mode="after")
    def check_dim(self):
        if self.dim_out % 2 != 0:
            raise ValueError("dim_out must be even")
        return self


class GaussianConfig(BaseModel):
    # To differentiate fourier and gaussian with pydantic
    name: Literal["gaussian"] = "gaussian"
    dim_in: int = 2
    dim_out: int = 256
    pos_scale: float = 10.0

    @model_validator(mode="after")
    def check_dim(self):
        if self.dim_out % 2 != 0:
            raise ValueError("dim_out must be even")
        return self


class CoordXConfig(BaseModel):
    # TODO: Abstract to any MLP
    name: Literal["coordx"] = "coordx"
    dim_in: int = 2
    dim_out: int = 256
    net_cfg: "MLPConfig | CudaMLPConfig"
    fusion_op: Literal["+", "*", "mean"] = "*"

    @model_validator(mode="after")
    def check_dim(self):
        dim_in = self.net_cfg.pos_encode_cfg.dim_in
        if dim_in != 1:
            raise ValueError(f"CoordX's positional encoder's dim_in != 1, got {dim_in}")
        return self


class CudaHashgridConfig(BaseModel):
    name: Literal["cudahashgrid"] = "cudahashgrid"
    dim_in: int = 2
    n_features_per_level: Literal[1, 2, 4, 8] = 2
    dim_out: int = 32
    log2_hashmap_size: int = 19
    base_resolution: int = 16
    per_level_scale: float = 2.0
    pos_dtype: Literal["float32", "float16"] = "float32"
    interpolation: Literal["Linear", "Nearest", "Smoothstep"] = "Linear"
    hash_grid_gauss_init: bool = False
    hash_grid_init_mean: float = 0.0
    hash_grid_init_std: float = 0.01

    @model_validator(mode="after")
    def check_dim(self):
        if self.dim_out % self.n_features_per_level != 0:
            raise ValueError("dim_out must be a multiple of n_features_per_level")
        n_levels = self.dim_out // self.n_features_per_level
        if n_levels > 128:
            raise ValueError(f"n_levels is too large: {n_levels}>128")
        return self

    def tcnn_config(self) -> dict[str, "Any"]:
        return dict(
            otype="Grid",
            type="Hash",
            n_levels=self.dim_out // self.n_features_per_level,
            n_features_per_level=self.n_features_per_level,
            log2_hashmap_size=self.log2_hashmap_size,
            base_resolution=self.base_resolution,
            per_level_scale=self.per_level_scale,
            interpolation=self.interpolation,
        )


# TODO: AdaIn encoder
PosEncoderType = (
    CoordXConfig
    | FourierConfig
    | GaussianConfig
    | NeRFConfig
    | CudaHashgridConfig
    | NoPosEncode
)


class MLPConfig(BaseModel):
    name: Literal["mlp"] = "mlp"
    num_layers: int
    dim_hidden: int = 256
    activation: ActivationType = "relu"
    final_activation: ActivationType = "none"
    use_bias: bool = True
    random_projection: bool = False
    pos_encode_cfg: PosEncoderType = Field(discriminator="name")
    bottleneck: bool = False
    bottleneck_override: list[int] | None = None


class CudaMLPConfig(BaseModel):
    # Don't use this, torch.compile is better than this anyway
    name: Literal["cudamlp"] = "cudamlp"
    num_layers: int
    dim_hidden: int = 256
    activation: ActivationType = "relu"
    final_activation: ActivationType = "none"
    use_bias: bool = True
    pos_encode_cfg: PosEncoderType = Field(discriminator="name")
    bottleneck: bool = False  # unused
    bottleneck_override: list[int] | None = None  # unused


class SirenNeRVConfig(BaseModel):
    # Unused
    name: Literal["sirennerv"] = "sirennerv"
    nerv_act: ActivationType = "gelu"
    up_sample: int = 2
    expand_ch: int = 16
    num_freq: int = 100
    num_layers: int
    dim_hidden: int = 256
    activation: ActivationType = "relu"
    final_activation: ActivationType = "none"
    use_bias: bool = True
    pos_encode_cfg: PosEncoderType = Field(discriminator="name")


LoraType = Literal["lora", "sinlora", "group", "no_lora"]
# Skip connection for every applicable layer in decoder
SkipConnectionType = Literal[None, "+", "*"]


class STRAINERNetConfig(BaseModel):
    name: Literal["strainer"] = "strainer"
    num_decoder_layers: int
    bottleneck_decoder: bool = False
    bottleneck_decoder_override: list[int] | None = None
    sep_patch_pix: bool = False  # ensure model dim is divisible by patch_size^2
    skip_connection: SkipConnectionType = None
    mlp_cfg: MLPConfig | CudaMLPConfig
    pos_encode_cfg: PosEncoderType = NoPosEncode()
    lora_type: LoraType = "sinlora"
    lora_rank: int = 1
    lora_omega: float = 30.0  # only applicable to sinlora
    incode: bool = False


NetConfig = MLPConfig | CudaMLPConfig | SirenNeRVConfig | STRAINERNetConfig

# Don't use box_cox, EXTREMELY slow. sym_power also doesn't work well.
TransformType = Literal["min_max", "z_score", "sym_power", "box_cox"]
DenoisingType = Literal["all_white", "all_black", "salt_pepper", "gaussian", "random"]


class TransformConfig(BaseModel):
    transform: TransformType = "min_max"
    normalization_range: tuple[float, float] = (0.0, 1.0)
    pn_beta: float = 0.05
    gamma_boundary: float = 5
    pn_k: float = 256.0
    gamma_trans: float = -1.0
    pn_cum: float = 0.5
    pn_beta: float = 0.05
    pn_buffer: float = 1.0
    pn_alpha: float = 0.01
    box_shift: float = 0.1
    denoising: bool = False
    denoising_type: DenoisingType = "gaussian"


class EncoderConfig(BaseModel):
    net: NetConfig = Field(discriminator="name")
    resize: tuple[int, int] | None = None
    yuv_size: tuple[int, int] | None = None
    patch_size: int | None = None
    patch_scales: list[Fraction] | None = None
    strided_patches: bool = False
    offset_patch_training: bool = True  # should be False...
    compile: bool = False  # torch.compile
    normalize_range: tuple[float, float] = (-1, 1)
    image_transform: TransformConfig = TransformConfig()
    inference_batch_size: int = 2  # How many decoders to run at once
    # Chunk x, y coordinates. This should be set high (4, 8, 16, ...)
    inference_chunk_size: int = 1


# Quanto not currently working
QuantMethod = Literal["post", "torchao", "quanto", "hqq", "bnb"]


class QuantizationConfig(BaseModel):
    quant_method: QuantMethod = "post"
    quant_bit: int = 8
    quanto_dtype: str = "qint8"
    quantize: bool = False
    qfloat: bool = False  # Quantize using torch's float8
    entropy_coding: CodingType = "huffman"
    keys_to_ignore: list[str] = []
    quant_axis: tuple = ()
    sparsity: float | None = None
    qat: bool = False
    ffnerv_qat: bool = False  # doesn't work


class TrainerConfig(BaseModel):
    # Not Used
    name: Literal["trainer"]
    num_iters: int = 5000
    lr: float = 3e-4
    lr_scheduler: LRSchedulerType | None = None
    group_size: int = 1
    scheduler_params: dict | None = None
    optimizer: OptimizerType = "adam"
    betas: tuple[float, float] = (0.9, 0.999)
    skip_save: bool = False
    skip_save_model: bool = False
    num_save_images: int = 2
    losses: list[tuple[LossType, float]] = [("mse", 1)]
    precision: Literal["bfloat16", "fp32", "fp16"] = "bfloat16"
    sampling: SamplingType = None
    edge_d: float = 1.0
    edge_delta: float = 0.1
    sampling_warmup: int = 0
    coord_sample_frac: float = 1.0
    eval_interval: int = 10
    save_interval: int = 500


# False trains the decoders the same as the shared encoder
# True freezes the encoder but has full separate decoders
# LoraFull does not work
# LoraDec is SIEDD default
AmortizedType = Literal[False, True, "LoraFull", "LoraDec"]


class StrainerConfig(BaseModel):
    name: Literal["strainer"]
    meta_frames: int = 10  # held equal to group_size
    iters: int = 2100
    shared_iters: int = 5001
    amortized: AmortizedType = False
    lora_dec_transfer_decoder: bool = False
    shared_lr: float = 1e-4
    lr: float = 1e-4
    lr_warmup: int = 100
    group_size: int = 10  # held equal to meta_frames
    lr_scheduler: LRSchedulerType | None = None
    scheduler_params: dict[str, Any] | None = None
    optimizer: OptimizerType = "adam"
    betas: tuple[float, float] = (0.9, 0.999)
    losses: list[tuple[LossType, float]] = [("mse", 1)]
    precision: Literal["bfloat16", "fp32", "fp16"] = "bfloat16"
    sampling: SamplingType = None
    edge_d: float = 1.0  # edge softmax temperature
    edge_delta: float = 0.1
    sampling_warmup: int = 0
    coord_sample_frac: float = 1.0  # Coordinate sampling proportion
    skip_save: bool = False
    skip_save_model: bool = False
    num_save_images: int = 2
    eval_interval: int = 10  # Set to >1000, or wall time will be insane
    save_interval: int = 500
    shared_encoder_path: Path | None = None  # For experiment 52


class RunConfig(BaseModel):
    encoder_cfg: EncoderConfig
    trainer_cfg: TrainerConfig | StrainerConfig
    quant_cfg: QuantizationConfig

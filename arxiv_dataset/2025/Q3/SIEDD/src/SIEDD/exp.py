from .configs import (
    RunConfig,
    StrainerConfig,
    QuantizationConfig,
    EncoderConfig,
    MLPConfig,
    STRAINERNetConfig,
    NeRFConfig,
    SineConfig,
    LoraType,
    OptimizerType,
    FourierConfig,
    CudaHashgridConfig,
    ActivationType,
    PosEncoderType,
    NoPosEncode,
    LRSchedulerType,
    TransformType,
    DenoisingType,
)
import pydantic_yaml
from pathlib import Path
import subprocess
import os
import sys

EXECUTOR = "SLURM"
PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent
EXPERIMENT_SCRATCH = PROJECT_ROOT / "experiment_scratch"
EXPERIMENT_SCRATCH.mkdir(exist_ok=True)
UVG = (
    "Beauty",
    "Jockey",
    "Bosphorus",
    "HoneyBee",
    "ReadySetGo",
    "ShakeNDry",
    "YachtRide",
)
USTC = (
    "ParkWalking",
    "Badminton",
    "FourPeople",
    "ShakingHands",
    "BasketballDrill",
    "Running",
    "Snooker",
    "Dancing",
    "BicycleDriving",
    "BasketballPass",
)
DAVIS = (
    "blackswan",
    "bmx-trees",
    "boat",
    "breakdance",
    "camel",
    "car-roundabout",
    "car-shadow",
    "cows",
    "dance-twirl",
    "dog",
)
YOUTUBE_8M = ("MarioKart",)
DATASET = os.getenv("DATASET")
if DATASET in (None, "UVG_1080p"):
    DATASET_PATH = "UVG"
    VIDEOS = UVG
    RES = "_1080p"
if DATASET == "UVG_4k":
    DATASET_PATH = "UVG_4k"
    VIDEOS = UVG
    RES = "_4k"
if DATASET == "Bunny":
    DATASET_PATH = "UVG_Bunny"
    VIDEOS = ("Bunny",)
    RES = "_720p"
if DATASET == "YOUTUBE_8M":
    DATASET_PATH = "YOUTUBE_8M"
    VIDEOS = YOUTUBE_8M
    RES = ""
if DATASET == "UVG_1080p_ALL":
    DATASET_PATH = "UVG"
    VIDEOS = ("",)
    RES = ""
if DATASET == "USTC_1080p":
    DATASET_PATH = "USTC"
    VIDEOS = USTC
    RES = "_1080p"
if DATASET == "DAVIS":  # 1080p only
    DATASET_PATH = "DAVIS"
    VIDEOS = DAVIS
    RES = ""
if DATASET == "DBG":
    DATASET_PATH = "UVG"
    VIDEOS = ("Jockey",)
    RES = "_1080p"
if DATASET == "DBG_4k":
    DATASET_PATH = "UVG_4k"
    VIDEOS = ("ReadySetGo",)
    RES = "_4k"


class StrainerEncoderConfig(EncoderConfig):
    net: STRAINERNetConfig  # pyright: ignore[reportIncompatibleVariableOverride]


class StrainerRunConfig(RunConfig):
    encoder_cfg: StrainerEncoderConfig  # pyright: ignore[reportIncompatibleVariableOverride]
    trainer_cfg: StrainerConfig  # pyright: ignore[reportIncompatibleVariableOverride]


def start_slurm(cfg: RunConfig, data_path: Path, name: str):
    from time import sleep

    cfg_path = EXPERIMENT_SCRATCH / f"exp_cfg_{name}.yaml"
    slurm_path = EXPERIMENT_SCRATCH / f"slurm_exp_cfg_{name}.sh"
    pydantic_yaml.to_yaml_file(cfg_path, cfg)
    args = f"--cfg {cfg_path} --data_path {data_path} --name {name}"
    cmd = f"uv run --env-file .env train {args}"
    exp_num = "_".join(name.split("_")[:2])  # name is exp_n_...
    cmd = f"export WANDB_GROUP={exp_num}\n{cmd}"

    slurm_template = (PROJECT_ROOT / "slurm_template.sh").read_text()
    slurm = slurm_template.replace("{{PROJECT_ROOT}}", str(PROJECT_ROOT))
    slurm = slurm.replace("{{CMD}}", cmd)
    slurm_path.write_text(slurm)
    os.chdir(EXPERIMENT_SCRATCH)
    subprocess.run(["sbatch", str(slurm_path)])
    sleep(3)


RUNPOD_RUN_CFGS: list[str] = []
RUNPOD_DATA_PATHS: list[str] = []
RUNPOD_RUN_NAMES: list[str] = []


def append_runpod(cfg: RunConfig, data_path: Path, name: str):
    cfg_path = EXPERIMENT_SCRATCH / f"exp_cfg_{name}.yaml"
    pydantic_yaml.to_yaml_file(cfg_path, cfg)
    relative_cfg_path = cfg_path.relative_to(PROJECT_ROOT)
    relative_data_path = data_path.relative_to(PROJECT_ROOT)
    RUNPOD_RUN_CFGS.append(str(relative_cfg_path))
    RUNPOD_DATA_PATHS.append(str(relative_data_path))
    RUNPOD_RUN_NAMES.append(name)


def start_run(cfg: RunConfig, data_path: Path, name: str):
    if EXECUTOR == "SLURM":
        start_slurm(cfg, data_path, name)
    elif EXECUTOR == "RUNPOD":
        append_runpod(cfg, data_path, name)


def default_strainer_config():
    cfg = StrainerRunConfig(
        encoder_cfg=StrainerEncoderConfig(
            yuv_size=(1080, 1920),
            patch_size=None,
            compile=True,
            net=STRAINERNetConfig(
                lora_omega=200,
                num_decoder_layers=2,
                mlp_cfg=MLPConfig(
                    num_layers=5,
                    activation=SineConfig(),
                    dim_hidden=128,
                    pos_encode_cfg=NeRFConfig(dim_out=18),
                ),
            ),
        ),
        trainer_cfg=StrainerConfig(
            amortized=True,
            name="strainer",
            sampling="uniform",
            precision="fp16",
            optimizer="schedule_free",
            coord_sample_frac=1.0 / 16.0,
            shared_iters=10001,
            lr=1e-4,
            iters=10001,
            group_size=10,
            meta_frames=10,
            eval_interval=1000,
            save_interval=1000,
        ),
        quant_cfg=QuantizationConfig(quant_bit=8, quantize=True),
    )
    return cfg


def default_strainer_config_lora_dec_512():
    cfg = default_strainer_config()
    cfg.trainer_cfg.group_size = 30
    cfg.trainer_cfg.meta_frames = 30
    cfg.trainer_cfg.coord_sample_frac = 1 / 64
    cfg.trainer_cfg.amortized = "LoraDec"
    cfg.encoder_cfg.net.mlp_cfg.num_layers = 5
    cfg.encoder_cfg.net.mlp_cfg.dim_hidden = 512
    cfg.encoder_cfg.net.num_decoder_layers = 2
    return cfg


def stable_config_3():
    cfg = default_strainer_config_lora_dec_512()
    cfg.trainer_cfg.optimizer = "schedule_free"
    cfg.trainer_cfg.coord_sample_frac = 9 / 1024
    cfg.encoder_cfg.patch_size = 3
    cfg.encoder_cfg.net.mlp_cfg.num_layers = 1
    cfg.encoder_cfg.net.mlp_cfg.dim_hidden = 768
    cfg.encoder_cfg.net.num_decoder_layers = 3
    return cfg


def stable_config_4():
    cfg = stable_config_3()
    cfg.encoder_cfg.net.lora_type = "no_lora"
    cfg.encoder_cfg.patch_size = None
    cfg.encoder_cfg.inference_batch_size = 1
    cfg.encoder_cfg.inference_chunk_size = 8
    cfg.trainer_cfg.coord_sample_frac = 1 / 1024
    cfg.encoder_cfg.image_transform.transform = "z_score"
    cfg.trainer_cfg.group_size = 20
    cfg.trainer_cfg.meta_frames = 20
    cfg.trainer_cfg.shared_iters = 20001
    cfg.trainer_cfg.iters = 20001
    return cfg


def SIEDD_M():
    cfg = stable_config_4()
    cfg.trainer_cfg.eval_interval = 20000
    cfg.trainer_cfg.save_interval = 20000
    cfg.quant_cfg.quant_method = "hqq"
    cfg.quant_cfg.quant_bit = 6
    return cfg


def SIEDD_S():
    cfg = SIEDD_M()
    cfg.encoder_cfg.net.mlp_cfg.dim_hidden = 512
    return cfg


def SIEDD_L():
    cfg = SIEDD_M()
    cfg.encoder_cfg.net.mlp_cfg.dim_hidden = 1024
    return cfg


def experiment_1():
    # Baseline
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = default_strainer_config()
        start_run(cfg, pth, f"exp_1_{d}")


def experiment_2():
    # Group size ablation
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for group_size in 5, 10, 40, 50, 100:
            cfg = default_strainer_config_lora_dec_512()

            cfg.trainer_cfg.optimizer = "schedule_free"
            cfg.trainer_cfg.coord_sample_frac = 9 / 1024
            cfg.encoder_cfg.patch_size = 3
            cfg.trainer_cfg.group_size = group_size
            cfg.trainer_cfg.meta_frames = group_size
            start_run(cfg, pth, f"exp_2_{d}_gs_{group_size}")


def experiment_3():
    # Bottleneck (Not Currently Used)
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = default_strainer_config()
        assert isinstance(cfg.encoder_cfg.net.mlp_cfg, MLPConfig)
        cfg.encoder_cfg.net.mlp_cfg.bottleneck = True
        start_run(cfg, pth, f"exp_3_{d}_btlnck")


def experiment_4():
    # Model size ablation
    combos = [(7, 128, 2), (5, 256, 2), (5, 512, 2), (5, 128, 4)]
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for i, (nl, dh, dl) in enumerate(combos):
            cfg = default_strainer_config()
            cfg.encoder_cfg.net.mlp_cfg.num_layers = nl
            cfg.encoder_cfg.net.mlp_cfg.dim_hidden = dh
            cfg.encoder_cfg.net.num_decoder_layers = dl
            start_run(cfg, pth, f"exp_4_{d}_combo_{i}")


def experiment_5():
    # LR search (lora)
    lrs = [1e-5, 1e-3, 1e-2]
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for lr in lrs:
            cfg = default_strainer_config_lora_dec_512()
            cfg.trainer_cfg.lr = lr
            cfg.trainer_cfg.shared_lr = lr
            start_run(cfg, pth, f"exp_5_{d}_lr_{lr:.0e}")


def experiment_6():
    # Model size search with shared decoder/lora
    combos = [(5, 128, 2), (5, 256, 2), (5, 512, 2)]
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for i, (nl, dh, dl) in enumerate(combos):
            cfg = default_strainer_config_lora_dec_512()
            cfg.encoder_cfg.net.mlp_cfg.num_layers = nl
            cfg.encoder_cfg.net.mlp_cfg.dim_hidden = dh
            cfg.encoder_cfg.net.num_decoder_layers = dl
            start_run(cfg, pth, f"exp_6_{d}_combo_{i}")


def experiment_7():
    # Lora rank search
    lora_ranks = [2, 4, 8, 16]
    lora_types: list[LoraType] = ["lora", "sinlora", "group"]
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for lora_type in lora_types:
            for lora_rank in lora_ranks:
                cfg = default_strainer_config_lora_dec_512()
                cfg.encoder_cfg.net.lora_rank = lora_rank
                cfg.encoder_cfg.net.lora_type = lora_type
                start_run(cfg, pth, f"exp_7_{d}_{lora_type}_rank_{lora_rank}")


def experiment_8():
    # Lora omega search
    lora_omegas = [100, 400, 800]
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for lora_omega in lora_omegas:
            cfg = default_strainer_config_lora_dec_512()
            cfg.encoder_cfg.net.lora_omega = lora_omega
            start_run(cfg, pth, f"exp_8_{d}_omega_{lora_omega}")


def experiment_9():
    # Check transferred decoder weights
    for d in VIDEOS:
        for lr in 1e-5, 5e-5:
            pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
            cfg = stable_config_3()
            cfg.trainer_cfg.lora_dec_transfer_decoder = True
            cfg.trainer_cfg.lr = lr
            start_run(cfg, pth, f"exp_9_{d}_{lr:.0e}")


def experiment_10():
    # sampling rate search
    sampling_rates = [1 / 128, 1 / 256, 1 / 512, 1 / 1024, 1 / 2048]
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for sr in sampling_rates:
            cfg = default_strainer_config_lora_dec_512()

            # No patch size
            cfg.encoder_cfg.patch_size = None
            cfg.trainer_cfg.coord_sample_frac = sr
            int_sr = int(1 / sr)
            # start_run(cfg, pth, f"exp_10_{d}_sr_1_{int_sr}")

            # patch size = 3
            cfg.encoder_cfg.patch_size = 3
            cfg.trainer_cfg.coord_sample_frac = sr * 9
            start_run(cfg, pth, f"exp_10_{d}_sr_9_{int_sr}")


def experiment_11():
    # Patch size testing
    patch_sizes = [3, 6, 15]
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for patch_size in patch_sizes:
            cfg = default_strainer_config_lora_dec_512()
            sample_frac = min(1.0, patch_size * patch_size / 64)
            cfg.trainer_cfg.coord_sample_frac = sample_frac
            cfg.encoder_cfg.patch_size = patch_size
            start_run(cfg, pth, f"exp_11_{d}_ps_{patch_size}")


def experiment_12():
    # Optimizer tests
    optims: list[OptimizerType] = ["grokadamw", "schedule_free", "adam"]
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for optim in optims:
            cfg = default_strainer_config_lora_dec_512()
            if optim == "muon":
                cfg.trainer_cfg.precision = "bfloat16"
            cfg.trainer_cfg.coord_sample_frac = 9 / 512
            cfg.encoder_cfg.patch_size = 3
            cfg.trainer_cfg.optimizer = optim
            start_run(cfg, pth, f"exp_12_{d}_{optim}")


def experiment_13():
    # Quantization testing
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = default_strainer_config_lora_dec_512()
        cfg.quant_cfg.quantize = False
        cfg.trainer_cfg.optimizer = "schedule_free"
        cfg.trainer_cfg.coord_sample_frac = 9 / 1024
        cfg.encoder_cfg.patch_size = 3
        cfg.encoder_cfg.net.mlp_cfg.num_layers = 1
        cfg.encoder_cfg.net.mlp_cfg.dim_hidden = 1024
        cfg.encoder_cfg.net.num_decoder_layers = 3
        start_run(cfg, pth, f"exp_13_{d}")


def experiment_14():
    # fourier + relu and hashgrid testing
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = default_strainer_config_lora_dec_512()
        cfg.trainer_cfg.optimizer = "schedule_free"
        cfg.trainer_cfg.coord_sample_frac = 9 / 512
        cfg.encoder_cfg.patch_size = 3

        for activation in "relu", "sine":
            for pos_enc in ("hash",):
                act: ActivationType = "relu"
                if activation == "sine":
                    act = SineConfig()

                if pos_enc == "fr":
                    enc: PosEncoderType = FourierConfig(dim_out=512)
                elif pos_enc == "hash":
                    cfg.encoder_cfg.compile = False
                    enc = CudaHashgridConfig()
                else:
                    enc = NoPosEncode()
                cfg.encoder_cfg.net.mlp_cfg.pos_encode_cfg = enc
                cfg.encoder_cfg.net.mlp_cfg.activation = act
                start_run(cfg, pth, f"exp_14_{d}_{pos_enc}_{activation}")


def experiment_15():
    # siren omega and finer
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = default_strainer_config_lora_dec_512()
        cfg.trainer_cfg.optimizer = "schedule_free"
        cfg.trainer_cfg.coord_sample_frac = 9 / 512
        cfg.encoder_cfg.patch_size = 3

        for use_finer in (False,):
            for omega in [40, 60, 80]:
                act = SineConfig(w0=omega, finer=use_finer)
                cfg.encoder_cfg.net.mlp_cfg.activation = act
                act_name = "finer" if use_finer else "siren"
                start_run(cfg, pth, f"exp_15_{d}_{act_name}_{omega}")


def experiment_16():
    # LR scheduler
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = default_strainer_config_lora_dec_512()
        cfg.trainer_cfg.optimizer = "schedule_free"
        cfg.trainer_cfg.coord_sample_frac = 9 / 512
        cfg.encoder_cfg.patch_size = 3

        schedulers: list[tuple[LRSchedulerType, dict[str, float | int]]] = [
            ("LinearLR", {"start_factor": 1.0, "end_factor": 0.5}),
            ("LinearLR", {"start_factor": 1.0, "end_factor": 0.1}),
            ("CyclicLR", {"base_lr": 1e-5, "max_lr": 1e-4, "step_size_up": 2000}),
        ]
        for i, (sched, params) in enumerate(schedulers):
            cfg.trainer_cfg.lr_scheduler = sched
            cfg.trainer_cfg.scheduler_params = params
            start_run(cfg, pth, f"exp_16_{d}_{sched}_{i}")


def experiment_17():
    # edge sampling
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = default_strainer_config_lora_dec_512()
        cfg.trainer_cfg.optimizer = "schedule_free"
        cfg.trainer_cfg.coord_sample_frac = 9 / 512
        cfg.encoder_cfg.patch_size = 3

        for delta in [1.0, 0.5, 0.1]:
            cfg.trainer_cfg.sampling = "edge"
            cfg.trainer_cfg.edge_d = 1.0
            cfg.trainer_cfg.edge_delta = delta
            start_run(cfg, pth, f"exp_17_{d}_{delta}")


def experiment_18():
    # Stable config
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = default_strainer_config_lora_dec_512()
        cfg.trainer_cfg.optimizer = "schedule_free"
        cfg.trainer_cfg.coord_sample_frac = 9 / 1024
        cfg.encoder_cfg.patch_size = 3

        start_run(cfg, pth, f"exp_18_{d}")


def experiment_19():
    # Model size ablation, with patching
    # combos = [(3, 512, 2), (4, 512, 2), (5, 512, 2), (5, 512, 3), (5, 512, 4)]
    # combos = [(1, 512, 2), (2, 512, 2), (3, 640, 3), (5, 640, 3)]
    # combos = [(1, 640, 3), (2, 640, 3)]
    # combos = [(0, 768, 3), (1, 768, 3), (1, 768, 4)]
    # combos = [(1, 896, 3), (1, 1024, 3)]
    combos = [(1, 768, 4), (1, 768, 5), (1, 768, 6)]
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for i, (nl, dh, dl) in enumerate(combos):
            cfg = default_strainer_config_lora_dec_512()
            cfg.trainer_cfg.optimizer = "schedule_free"
            cfg.trainer_cfg.coord_sample_frac = 9 / 1024
            cfg.encoder_cfg.patch_size = 3
            cfg.encoder_cfg.net.mlp_cfg.num_layers = nl
            cfg.encoder_cfg.net.mlp_cfg.dim_hidden = dh
            cfg.encoder_cfg.net.num_decoder_layers = dl
            start_run(cfg, pth, f"exp_19_{d}_combo_{i}")


def experiment_20():
    # Strided patching
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = stable_config_3()
        cfg.encoder_cfg.strided_patches = True

        start_run(cfg, pth, f"exp_20_{d}")


def experiment_21():
    # Offset patch training
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = stable_config_3()
        cfg.encoder_cfg.offset_patch_training = True

        start_run(cfg, pth, f"exp_21_{d}")


def experiment_22():
    # Coordinate normalization
    for norm_range in [(-2, 2), (-3, 3)]:
        for d in VIDEOS:
            pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
            cfg = stable_config_3()

            cfg.encoder_cfg.net.mlp_cfg.pos_encode_cfg = NoPosEncode()
            cfg.encoder_cfg.normalize_range = norm_range

            start_run(cfg, pth, f"exp_22_{d}_{norm_range[0]}_{norm_range[1]}")


def experiment_23():
    # Image transformations (symmetric power paper)
    methods: list[TransformType] = ["min_max", "z_score", "sym_power"]
    for method in methods:
        for d in VIDEOS:
            pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
            cfg = stable_config_3()

            cfg.encoder_cfg.image_transform.transform = method
            if method == "min_max":
                cfg.encoder_cfg.image_transform.normalization_range = (-1, 1)
                start_run(cfg, pth, f"exp_23_{d}_{method}_-1_1")
                cfg.encoder_cfg.image_transform.normalization_range = (-2, 2)
                start_run(cfg, pth, f"exp_23_{d}_{method}_-2_2")
            else:
                start_run(cfg, pth, f"exp_23_{d}_{method}")


def experiment_24():
    # fourier and nerf but with trainable parameters
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = stable_config_3()

        for pos_enc in ("fr", "nerf"):
            act = SineConfig()

            if pos_enc == "fr":
                enc: PosEncoderType = FourierConfig(dim_out=512, trainable=True)
            elif pos_enc == "nerf":
                enc = NeRFConfig(dim_out=18, trainable=True)
            else:
                raise ValueError()
            cfg.encoder_cfg.net.mlp_cfg.pos_encode_cfg = enc
            cfg.encoder_cfg.net.mlp_cfg.activation = act
            start_run(cfg, pth, f"exp_24_{d}_{pos_enc}")


def experiment_25():
    # no shared encoder
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = stable_config_3()
        cfg.encoder_cfg.net.mlp_cfg.num_layers = -1
        start_run(cfg, pth, f"exp_25_{d}")


def experiment_26():
    # deeper model with lower learning rate
    for d in VIDEOS:
        for lr in 1e-5, 5e-5:
            pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
            cfg = stable_config_3()
            cfg.encoder_cfg.net.num_decoder_layers = 5
            cfg.trainer_cfg.lr = lr
            cfg.trainer_cfg.shared_lr = lr
            start_run(cfg, pth, f"exp_26_{d}_{lr:.0e}")


def experiment_27():
    # Skip connections
    for d in VIDEOS:
        for num_layers in 3, 5:
            for skip in ("times",):
                skip_val = "*" if skip == "times" else skip
                pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
                cfg = stable_config_3()
                cfg.encoder_cfg.net.num_decoder_layers = num_layers
                cfg.encoder_cfg.net.skip_connection = skip_val
                start_run(cfg, pth, f"exp_27_{d}_{num_layers}_{skip}")


def experiment_28():
    # L1 Loss
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = stable_config_3()
        cfg.trainer_cfg.losses = [("l1", 1.0)]
        start_run(cfg, pth, f"exp_28_{d}")


def experiment_29():
    # rechecking no patch size
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = stable_config_3()
        cfg.encoder_cfg.patch_size = None
        cfg.encoder_cfg.inference_batch_size = 1
        cfg.trainer_cfg.coord_sample_frac = 1 / 1024
        start_run(cfg, pth, f"exp_29_{d}")
        cfg.encoder_cfg.net.mlp_cfg.dim_hidden = 864
        cfg.encoder_cfg.inference_chunk_size = 4
        start_run(cfg, pth, f"exp_29_{d}_big")
        cfg.encoder_cfg.net.bottleneck_decoder = True
        start_run(cfg, pth, f"exp_29_{d}_big_bottleneck")
        cfg.encoder_cfg.net.mlp_cfg.dim_hidden = 768
        start_run(cfg, pth, f"exp_29_{d}_bottleneck")


def experiment_30():
    # Last layer activation function
    for d in VIDEOS:
        for activation in "tanh", "sigmoid":  # , "sine":
            pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
            cfg = stable_config_3()
            if activation == "sine":
                for omega in 1.58, 30, 60:
                    cfg.encoder_cfg.net.mlp_cfg.final_activation = SineConfig(w0=omega)
                    cfg.encoder_cfg.image_transform.transform = "min_max"
                    cfg.encoder_cfg.image_transform.normalization_range = (-1, 1)
                    start_run(cfg, pth, f"exp_30_{d}_{activation}_{omega}")
                continue
            elif activation == "sigmoid":
                cfg.encoder_cfg.net.mlp_cfg.final_activation = "sigmoid"
                cfg.encoder_cfg.image_transform.transform = "min_max"
                cfg.encoder_cfg.image_transform.normalization_range = (0, 1)
            elif activation == "tanh":
                cfg.encoder_cfg.net.mlp_cfg.final_activation = "tanh"
                cfg.encoder_cfg.image_transform.transform = "min_max"
                cfg.encoder_cfg.image_transform.normalization_range = (-1, 1)
            start_run(cfg, pth, f"exp_30_{d}_{activation}")


def experiment_31():
    # LinearSine
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = stable_config_3()
        # cfg.encoder_cfg.net.mlp_cfg.activation = SineConfig(linear=True)
        # start_run(cfg, pth, f"exp_31_{d}")
        cfg.encoder_cfg.net.mlp_cfg.activation = SineConfig(linear=True, power=3.0)
        start_run(cfg, pth, f"exp_31_{d}_pow3")
        cfg.encoder_cfg.net.mlp_cfg.activation = SineConfig(finer=True)
        start_run(cfg, pth, f"exp_31_{d}_finer")


def experiment_32():
    # recheck high sampling rate
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = stable_config_3()
        cfg.trainer_cfg.coord_sample_frac = 9 / 32
        start_run(cfg, pth, f"exp_32_{d}")


def experiment_33():
    # NeRF dim size
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = stable_config_3()
        cfg.encoder_cfg.net.mlp_cfg.pos_encode_cfg = NeRFConfig(
            dim_out=64, include_coord=False
        )
        start_run(cfg, pth, f"exp_33_{d}")
        cfg.encoder_cfg.net.mlp_cfg.pos_encode_cfg = NeRFConfig(
            dim_out=66, include_coord=True
        )
        start_run(cfg, pth, f"exp_33_{d}_ic")
        cfg.encoder_cfg.net.mlp_cfg.pos_encode_cfg = NeRFConfig(
            dim_out=34, include_coord=True
        )
        start_run(cfg, pth, f"exp_33_{d}_34_ic")


def experiment_34():
    # Search for good cuda hashgrid params
    cfgs: list[CudaHashgridConfig] = [
        CudaHashgridConfig(),
        CudaHashgridConfig(dim_out=64),
        CudaHashgridConfig(hash_grid_gauss_init=True),
        CudaHashgridConfig(per_level_scale=1.5),
        CudaHashgridConfig(log2_hashmap_size=15),
        CudaHashgridConfig(n_features_per_level=4),
        CudaHashgridConfig(hash_grid_gauss_init=True, hash_grid_init_std=0.1),
        CudaHashgridConfig(hash_grid_gauss_init=True, hash_grid_init_std=0.5),
        CudaHashgridConfig(hash_grid_gauss_init=True, hash_grid_init_std=1.0),
        CudaHashgridConfig(hash_grid_gauss_init=True),
        CudaHashgridConfig(hash_grid_gauss_init=True, dim_out=128),
    ]
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for i, grid_cfg in enumerate(cfgs):
            if i < 10:
                continue
            cfg = stable_config_3()
            if i == 9:
                cfg.encoder_cfg.normalize_range = (0, 1)
            cfg.encoder_cfg.net.mlp_cfg.pos_encode_cfg = grid_cfg
            start_run(cfg, pth, f"exp_34_{d}_{i}")


def experiment_35():
    # no lora
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = stable_config_3()
        cfg.encoder_cfg.net.lora_type = "no_lora"
        start_run(cfg, pth, f"exp_35_{d}")
        cfg.encoder_cfg.patch_size = None
        cfg.encoder_cfg.inference_batch_size = 1
        cfg.trainer_cfg.coord_sample_frac = 1 / 1024
        # start_run(cfg, pth, f"exp_35_{d}_no_ps")


def experiment_36():
    # full test run (No patch)
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = stable_config_3()
        cfg.encoder_cfg.net.lora_type = "no_lora"
        cfg.encoder_cfg.patch_size = None
        cfg.encoder_cfg.inference_batch_size = 1
        cfg.trainer_cfg.coord_sample_frac = 1 / 1024
        cfg.encoder_cfg.inference_chunk_size = 8
        cfg.encoder_cfg.image_transform.transform = "z_score"
        # start_run(cfg, pth, f"exp_36_{d}")
        cfg.trainer_cfg.group_size = 20
        cfg.trainer_cfg.meta_frames = 20
        cfg.trainer_cfg.shared_iters = 20001
        # start_run(cfg, pth, f"exp_36_{d}_2")
        cfg.trainer_cfg.iters = 20001
        # 36f's config is equivalent to stable_config_4()
        start_run(cfg, pth, f"exp_36f_{DATASET_PATH}_{d}")


def experiment_37():
    # separate patch pixels
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = stable_config_3()
        cfg.encoder_cfg.net.lora_type = "no_lora"
        cfg.encoder_cfg.image_transform.transform = "z_score"
        cfg.encoder_cfg.net.mlp_cfg.dim_hidden = 765  # divisible by p^2
        cfg.encoder_cfg.net.sep_patch_pix = True
        start_run(cfg, pth, f"exp_37_{d}")


def experiment_38():
    # Shared encoder across all of UVG
    pth = PROJECT_ROOT / "data" / "UVG"
    cfg = stable_config_4()
    start_run(cfg, pth, "exp_38_all")


def experiment_39():
    # more arch search
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = stable_config_4()
        cfg.encoder_cfg.net.mlp_cfg.dim_hidden = 512
        cfg.trainer_cfg.group_size = 12
        cfg.trainer_cfg.meta_frames = 12
        # start_run(cfg, pth, f"exp_39_{d}")
        cfg.encoder_cfg.net.mlp_cfg.dim_hidden = 1024
        cfg.trainer_cfg.group_size = 30
        cfg.trainer_cfg.meta_frames = 30
        start_run(cfg, pth, f"exp_39-2_{d}")


def experiment_40():
    # Baseline unquantized
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = SIEDD_S()
        cfg.quant_cfg.quantize = False
        start_run(cfg, pth, f"exp_40-S_{DATASET_PATH}_{d}")
        cfg = SIEDD_M()
        cfg.quant_cfg.quantize = False
        start_run(cfg, pth, f"exp_40-M_{DATASET_PATH}_{d}")
        cfg = SIEDD_L()
        cfg.quant_cfg.quantize = False
        start_run(cfg, pth, f"exp_40-L_{DATASET_PATH}_{d}")


def experiment_41():
    # Ablate number of decoder layers
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for num_decoder_layers in [2, 4, 5]:
            cfg = stable_config_4()
            cfg.encoder_cfg.net.num_decoder_layers = num_decoder_layers
            start_run(cfg, pth, f"exp_41-{num_decoder_layers}_{DATASET_PATH}_{d}")


def experiment_42():
    # Ablate model dim
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for model_dim in [256, 512, 1024]:
            cfg = stable_config_4()
            cfg.encoder_cfg.net.mlp_cfg.dim_hidden = model_dim
            start_run(cfg, pth, f"exp_42-{model_dim}_{DATASET_PATH}_{d}")


def experiment_43():
    # QAT
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for bit in 8, 7, 6:
            cfg = stable_config_4()
            cfg.quant_cfg.quant_bit = bit
            cfg.quant_cfg.qat = True
            cfg.quant_cfg.quant_axis = (-1,)
            # ignore last layer (decoderINR.net.3...) and biases
            cfg.quant_cfg.keys_to_ignore = ["3", "bias"]
            start_run(cfg, pth, f"exp_43-{bit}_{DATASET_PATH}_{d}")
            cfg.quant_cfg.qat = False
            cfg.quant_cfg.ffnerv_qat = True
            start_run(cfg, pth, f"exp_43-ff-{bit}_{DATASET_PATH}_{d}")


def experiment_44():
    # Bottleneck
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = stable_config_4()
        cfg.encoder_cfg.net.bottleneck_decoder = True
        start_run(cfg, pth, f"exp_44-bottleneck_{DATASET_PATH}_{d}")
        cfg.encoder_cfg.net.bottleneck_decoder_override = [768, 768, 768, 256, 3]
        # Bottleneck the penultimate layer
        start_run(cfg, pth, f"exp_44-penultbtlnck_{DATASET_PATH}_{d}")


def experiment_45():
    # UVG 4K Testing
    # combo: # iters, group size, model dim, patch size
    combos: list[tuple[int, int, int, int | None]] = [
        (20000, 20, 768, None),  # .07bpp
        (10000, 10, 768, None),  # .15bpp
        (10000, 10, 1024, None),  # .3bpp
        (10000, 10, 1024, 3),  # .3bpp ps 3
        (10000, 10, 1024, 6),  # .3bpp ps 6
    ]
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for i, (iters, group_size, model_dim, patch_size) in enumerate(combos):
            cfg = stable_config_4()
            cfg.encoder_cfg.inference_chunk_size = 32 if patch_size is None else 64
            cfg.trainer_cfg.eval_interval = iters
            cfg.trainer_cfg.iters = iters + 1
            cfg.trainer_cfg.meta_frames = group_size
            cfg.trainer_cfg.group_size = group_size
            cfg.encoder_cfg.net.mlp_cfg.dim_hidden = model_dim
            cfg.encoder_cfg.patch_size = patch_size
            if cfg.encoder_cfg.patch_size is not None:
                # Must quantize last layer or bpp will skyrocket
                cfg.quant_cfg.quant_method = "post"
                cfg.quant_cfg.quant_bit = 8
            else:
                cfg.quant_cfg.quant_method = "hqq"
                cfg.quant_cfg.quant_bit = 6
            frac = (patch_size**2) / 1024 if patch_size else 1 / 1024
            cfg.trainer_cfg.coord_sample_frac = frac
            start_run(cfg, pth, f"exp_45-{i}_{DATASET_PATH}_{d}")


def experiment_46():
    # Lora shared encoder's decoder transfer weights (ERRONEOUS EXPERIMENT)
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = stable_config_4()
        cfg.trainer_cfg.lora_dec_transfer_decoder = True
        start_run(cfg, pth, f"exp_46_{DATASET_PATH}_{d}")


def experiment_47():
    # Up # iterations to 50k
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = stable_config_4()
        cfg.trainer_cfg.shared_iters = 50001
        cfg.trainer_cfg.iters = 50001
        start_run(cfg, pth, f"exp_47_{DATASET_PATH}_{d}")


def experiment_48():
    # Shared encoder with low iters
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = stable_config_4()
        cfg.trainer_cfg.shared_iters = 1001
        # start_run(cfg, pth, f"exp_48_{DATASET_PATH}_{d}")
        cfg.trainer_cfg.shared_iters = 2001
        # start_run(cfg, pth, f"exp_48-2000_{DATASET_PATH}_{d}")
        cfg.trainer_cfg.shared_iters = 5001
        # start_run(cfg, pth, f"exp_48-5000_{DATASET_PATH}_{d}")
        cfg.trainer_cfg.shared_iters = 501
        cfg.trainer_cfg.eval_interval = 500
        cfg.trainer_cfg.save_interval = 500
        # start_run(cfg, pth, f"exp_48-500_{DATASET_PATH}_{d}")
        cfg.trainer_cfg.shared_iters = 51
        cfg.trainer_cfg.eval_interval = 50
        cfg.trainer_cfg.save_interval = 50
        start_run(cfg, pth, f"exp_48-50_{DATASET_PATH}_{d}")
        cfg.trainer_cfg.shared_iters = 21
        cfg.trainer_cfg.eval_interval = 20
        cfg.trainer_cfg.save_interval = 20
        start_run(cfg, pth, f"exp_48-20_{DATASET_PATH}_{d}")


def experiment_49():
    # Denoising
    denoising_types: list[DenoisingType] = [
        "all_white",
        "all_black",
        "salt_pepper",
        "gaussian",
        "random",
    ]
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for denoising_type in denoising_types:
            cfg = SIEDD_L()
            cfg.encoder_cfg.image_transform.denoising = True
            cfg.encoder_cfg.image_transform.denoising_type = denoising_type
            cfg.trainer_cfg.losses = [("l1", 1.0)]
            denoising_type_name = denoising_type.replace("_", "-")
            start_run(cfg, pth, f"exp_49-{denoising_type_name}_{DATASET_PATH}_{d}")


def experiment_50():
    # 720p (SIEDD S)
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = SIEDD_S()
        start_run(cfg, pth, f"exp_50_{DATASET_PATH}_{d}")


def experiment_51():
    # HD Patching
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for ps in [3, 6]:
            cfg = stable_config_4()
            cfg.encoder_cfg.patch_size = ps
            cfg.trainer_cfg.coord_sample_frac = (ps**2) / 1024
            cfg.quant_cfg.quant_method = "hqq"
            cfg.quant_cfg.quant_bit = 6
            # start_run(cfg, pth, f"exp_51-{ps}_{DATASET_PATH}_{d}")
            cfg.quant_cfg.quant_method = "post"
            cfg.quant_cfg.quant_bit = 8
            start_run(cfg, pth, f"exp_51-{ps}-post_{DATASET_PATH}_{d}")


def experiment_52():
    # Train DAVIS with UVG Shared Encoder
    shared_encoder_path = os.environ["SHARED_ENCODER_PATH"]
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = SIEDD_M()
        cfg.trainer_cfg.shared_encoder_path = Path(shared_encoder_path)
        start_run(cfg, pth, f"exp_52_{DATASET_PATH}_{d}")


def experiment_53():
    # SIEDD M
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = SIEDD_M()
        start_run(cfg, pth, f"exp_53_{DATASET_PATH}_{d}")


def experiment_54():
    # Group size ablation
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for group_size in [10, 15, 25, 30]:
            cfg = SIEDD_M()
            cfg.trainer_cfg.group_size = group_size
            cfg.trainer_cfg.meta_frames = group_size
            start_run(cfg, pth, f"exp_54-{group_size}_{d}")


def experiment_55():
    # SIEDD M LoRA
    lora_ranks = [4, 8]
    lora_types: list[LoraType] = ["lora", "sinlora"]
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        for lora_type in lora_types:
            for lora_rank in lora_ranks:
                cfg = SIEDD_M()
                cfg.quant_cfg.quant_method = "post"
                cfg.quant_cfg.quant_bit = 8
                cfg.quant_cfg.keys_to_ignore = ["3", "bias", ".U", ".V"]
                cfg.encoder_cfg.net.lora_rank = lora_rank
                cfg.encoder_cfg.net.lora_type = lora_type
                start_run(cfg, pth, f"exp_55-{lora_type}-{lora_rank}_{d}")


def experiment_56():
    # SIEDD L
    for d in VIDEOS:
        pth = PROJECT_ROOT / "data" / DATASET_PATH / f"{d}{RES}"
        cfg = SIEDD_L()
        start_run(cfg, pth, f"exp_56_{DATASET_PATH}_{d}")


EXPERIMENTS = [
    experiment_1,
    experiment_2,
    experiment_3,
    experiment_4,
    experiment_5,
    experiment_6,
    experiment_7,
    experiment_8,
    experiment_9,
    experiment_10,
    experiment_11,
    experiment_12,
    experiment_13,
    experiment_14,
    experiment_15,
    experiment_16,
    experiment_17,
    experiment_18,
    experiment_19,
    experiment_20,
    experiment_21,
    experiment_22,
    experiment_23,
    experiment_24,
    experiment_25,
    experiment_26,
    experiment_27,
    experiment_28,
    experiment_29,
    experiment_30,
    experiment_31,
    experiment_32,
    experiment_33,
    experiment_34,
    experiment_35,
    experiment_36,
    experiment_37,
    experiment_38,
    experiment_39,
    experiment_40,
    experiment_41,
    experiment_42,
    experiment_43,
    experiment_44,
    experiment_45,
    experiment_46,
    experiment_47,
    experiment_48,
    experiment_49,
    experiment_50,
    experiment_51,
    experiment_52,
    experiment_53,
    experiment_54,
    experiment_55,
    experiment_56,
]


def start():
    exps = " ".join(sys.argv[1:])
    experiments = list(map(int, exps.replace(" ", "").split(",")))
    for exp in experiments:
        if exp < 0 or exp > len(EXPERIMENTS):
            raise ValueError(f"Invalid experiment number: {exp}")
    print("Starting experiments", str(experiments)[1:-1], "on", DATASET)
    for exp in experiments:
        EXPERIMENTS[exp - 1]()


if __name__ == "__main__":
    start()

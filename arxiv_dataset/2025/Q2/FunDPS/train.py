import os
import re
import json
import random
import wandb
import torch
import dnnlib
from datetime import datetime
from torch_utils import distributed as dist
from training import training_loop
from utils.yaml_config import Config, process_arguments

import warnings

warnings.filterwarnings("ignore", "Grad strides do not match bucket view strides")  # False warning printed by PyTorch 1.12.

# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def main():
    # Load configuration
    args = process_arguments(default_conf="configs/training/default.yml", debug_conf="configs/training/debug.yml")
    conf = Config(args)

    torch.multiprocessing.set_start_method("spawn")
    dist.init()

    if conf["seed"] is None:
        seed = torch.randint(1 << 31, size=[], device=torch.device("cuda"))
        torch.distributed.broadcast(seed, src=0)
        conf.update("seed", int(seed))

    # Initialize wandb
    if dist.get_rank() == 0:
        wandb.init(
            config=conf.to_dict(),
            name=conf["name"],
            mode=conf["wandb"],
        )
        wandb.run.log_code(root=".")

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(
        class_name="training.dataset_hf.PDEDataset",
        path=conf["data"],
        resolution=conf["resolution"],
        use_labels=conf["cond"],
        xflip=conf["xflip"],
        cache=conf["cache"],
    )
    c.data_loader_kwargs = dnnlib.EasyDict(num_workers=conf["workers"], pin_memory=True, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", lr=conf["lr"], betas=[0.9, 0.999], eps=1e-8)
    c.sampler_kwargs = dnnlib.EasyDict(class_name="training.noise_samplers.RBFKernel", scale=conf["rbf_scale"])

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        c.dataset_kwargs.resolution = dataset_obj.resolution  # be explicit about dataset resolution
        c.dataset_kwargs.max_size = len(dataset_obj)  # be explicit about dataset size
        if conf["cond"] and not dataset_obj.has_labels:
            raise ValueError("--cond=True requires labels specified in dataset.json")
        del dataset_obj  # conserve memory
    except IOError as err:
        raise ValueError(f"--data: {err}")

    # Network architecture.
    if conf["arch"] == "ddpmpp":
        c.network_kwargs.update(model_type="SongUNet", embedding_type="positional", encoder_type="standard", decoder_type="standard")
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1, 1], model_channels=128, channel_mult=[2, 2, 2])
    elif conf["arch"] == "ncsnpp":
        c.network_kwargs.update(model_type="SongUNet", embedding_type="fourier", encoder_type="residual", decoder_type="standard")
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1, 3, 3, 1], model_channels=128, channel_mult=[2, 2, 2])
    elif conf["arch"] == "adm":
        c.network_kwargs.update(model_type="DhariwalUNet", model_channels=192, channel_mult=[1, 2, 3, 4])
    elif conf["arch"] == "ddpmpp-uno":
        c.network_kwargs.update(model_type="SongUNO", embedding_type="positional", encoder_type="standard", decoder_type="standard")
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1, 1], model_channels=128, channel_mult=[2, 2, 2])
        c.network_kwargs.update(
            cond=conf["cond"],
            attn_resolutions=conf["attn_resolutions"],
            num_blocks=conf["num_blocks"],
            fmult=conf["fmult"],
            rank=conf["rank"],
        )
    else:
        raise ValueError(f"Invalid architecture: {conf['arch']}")

    # Preconditioning & loss function.
    if conf["precond"] == "vp":
        c.network_kwargs.class_name = "training.networks.VPPrecond"
        c.loss_kwargs.class_name = "training.loss.VPLoss"
    elif conf["precond"] == "ve":
        c.network_kwargs.class_name = "training.networks.VEPrecond"
        c.loss_kwargs.class_name = "training.loss.VELoss"
    elif conf["precond"] == "edm":
        c.network_kwargs.class_name = "training.networks.EDMPrecond"
        c.loss_kwargs.class_name = "training.loss.EDMLossWithSampler" if conf["arch"] == "ddpmpp-uno" else "training.loss.EDMLoss"
    else:
        raise ValueError(f"Invalid preconditioning: {conf['precond']}")

    # Network options.
    if conf["cbase"] is not None:
        c.network_kwargs.model_channels = conf["cbase"]
    if conf["cres"] is not None:
        c.network_kwargs.channel_mult = conf["cres"]
    c.network_kwargs.update(dropout=conf["dropout"], use_fp16=conf["fp16"])
    if conf["nn_resolution"] is not None:
        assert conf["nn_resolution"] <= conf["resolution"]
        c.network_kwargs.update(img_resolution=conf["nn_resolution"])
    else:
        c.network_kwargs.update(img_resolution=conf["resolution"])

    # Training options.
    c.total_kimg = max(int(conf["duration"] * 1000), 1)
    c.lr_rampup_kimg = int(conf["lr_rampup"] * 1000)
    c.ema_halflife_kimg = int(conf["ema"] * 1000)
    c.update(batch_size=conf["batch"], batch_gpu=conf["batch_gpu"])
    c.update(loss_scaling=conf["ls"], cudnn_benchmark=conf["bench"])
    c.update(kimg_per_tick=conf["tick"], snapshot_ticks=conf["snap"], state_dump_ticks=conf["dump"])
    c.cond = conf["cond"]
    c.seed = conf["seed"]

    # Resume training.
    if conf["resume"]:
        match = re.fullmatch(r"training-state-(\d+).pt", os.path.basename(conf["resume"]))
        if not match or not os.path.isfile(conf["resume"]):
            raise ValueError("--resume must point to training-state-*.pt from a previous training run")
        c.resume_pkl = os.path.join(os.path.dirname(conf["resume"]), f"network-snapshot-{match.group(1)}.pkl")
        c.resume_nimg = int(match.group(1))
        c.resume_state_dump = conf["resume"]

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    else:
        formatted_time = datetime.fromtimestamp(wandb.run.start_time).strftime("%m%d_%H%M%S")
        desc = f"{formatted_time}-{conf['name']}-{wandb.run.id}"
        c.run_dir = os.path.join(conf["outdir"], desc)
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0("Training options:")
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f"Output directory:        {c.run_dir}")
    dist.print0(f"Dataset path:            {c.dataset_kwargs.path}")
    dist.print0(f"Class-conditional:       {c.dataset_kwargs.use_labels}")
    dist.print0(f"Network architecture:    {conf['arch']}")
    dist.print0(f"Preconditioning & loss:  {conf['precond']}")
    dist.print0(f"Number of GPUs:          {dist.get_world_size()}")
    dist.print0(f"Batch size:              {c.batch_size}")
    dist.print0(f"Mixed-precision:         {c.network_kwargs.use_fp16}")
    dist.print0()

    # Dry run?
    if conf["dry_run"]:
        dist.print0("Dry run; exiting.")
        return

    # Create output directory.
    dist.print0("Creating output directory...")
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, "training_options.json"), "wt") as f:
            json.dump(c, f, indent=2)

    # Train.
    training_loop.training_loop(**c)

    if dist.get_rank() == 0:
        wandb.finish()


if __name__ == "__main__":
    main()

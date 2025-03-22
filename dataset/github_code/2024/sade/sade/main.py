"""Training and evaluation"""
import logging
import os
import sys
import warnings

import ml_collections
import torch
import wandb
from absl import app, flags
from ml_collections.config_flags import config_flags
from run.eval import evaluator, segmentation_evaluator
from run.finetune import finetuner
from run.flows import flow_evaluator, flow_trainer
from run.inference import inference
from run.train import trainer

warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum(
    "mode",
    None,
    ["train", "finetune", "eval", "seg-eval", "score", "sweep", "flow-train", "flow-eval", "inference"],
    "Running mode: train or eval",
)
flags.DEFINE_string("eval_folder", "eval", "The folder name for storing evaluation results")
flags.DEFINE_string(
    "sweep_id", None, "Optional ID for a sweep controller if running a sweep."
)
flags.DEFINE_string("project", "sade", "Wandb project name.")
flags.DEFINE_bool("cuda_opt", False, "Whether to use cuda benchmark and tf32 matmul.")
# flags.DEFINE_string("pretrain_dir", None, "Directory with pretrained weights.")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def setup_logger(workdir):
    # Set logger so that it outputs to both console and file
    # Make logging work for both disk and network Storage
    gfile_stream = open(os.path.join(workdir, "stdout.txt"), "w")
    file_handler = logging.StreamHandler(gfile_stream)
    stdout_handler = logging.StreamHandler(sys.stdout)

    # Override root handler
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
        handlers=[file_handler, stdout_handler],
    )


def main(argv):
    if FLAGS.cuda_opt:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True

    if FLAGS.mode == "train":
        # Create the working directory
        os.makedirs(FLAGS.workdir, exist_ok=True)
        setup_logger(FLAGS.workdir)

        with wandb.init(
            project=FLAGS.project,
            config=FLAGS.config.to_dict(),
            resume="allow",
            sync_tensorboard=True,
            settings=wandb.Settings(code_dir="."),
        ):
            config = ml_collections.ConfigDict(wandb.config)

            # Use run id for directory name
            workdir = os.path.join(FLAGS.workdir, f"finetune", wandb.run.id)
            os.makedirs(workdir, exist_ok=True)
            setup_logger(workdir)

            # Run the training pipeline
            trainer(config, FLAGS.workdir)
    elif FLAGS.mode == "finetune":
        with wandb.init(
            project=FLAGS.project,
            config=FLAGS.config.to_dict(),
            resume="allow",
            sync_tensorboard=False,
            settings=wandb.Settings(code_dir="."),
        ):
            config = ml_collections.ConfigDict(wandb.config)

            # Use run id for directory name
            workdir = os.path.join(FLAGS.workdir, f"finetune", wandb.run.id)
            os.makedirs(workdir, exist_ok=True)
            setup_logger(workdir)

            finetuner(FLAGS.config, workdir)

    elif FLAGS.mode == "eval":
        evaluator(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "seg-eval":
        segmentation_evaluator(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "flow-train":
        # Create the working directory
        os.makedirs(FLAGS.workdir, exist_ok=True)

        # Override root handler
        logging.root.handlers = []
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
        )

        with wandb.init(
            project=FLAGS.project,
            config=FLAGS.config.to_dict(),
            resume="allow",
            sync_tensorboard=True,
            settings=wandb.Settings(code_dir="."),
        ):
            config = ml_collections.ConfigDict(wandb.config)

            # Run the training pipeline
            flow_trainer(config, FLAGS.workdir)
    elif FLAGS.mode == "flow-eval":
        flow_evaluator(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "inference":
        inference(FLAGS.config, FLAGS.workdir)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    app.run(main)

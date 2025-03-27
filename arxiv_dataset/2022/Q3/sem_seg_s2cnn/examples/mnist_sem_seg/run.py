#!/usr/bin/env python3

import argparse
import subprocess
import sys
import os
from pathlib import Path

import get_paths


def env_prefix(env):
    if env == "local":
        return []
    if env == "singularity":
        container_path = os.path.join(get_paths.get_container_path(), "s2cnn_container.sif")
        command = ["singularity", "exec", "--nv", "--no-home"]
        command += ["--env", "PYTHONPATH=$PYTHONPATH:" + get_paths.get_s2cnn_path()]
        for path in get_paths.get_bind_paths():
            command += ["--bind", path]
        command += [container_path]
        return command


def build_sin(run_args, sub_args):
    if sub_args != []:
        print(f"unknown arguments: {' '.join(sub_args)}")
        sys.exit(1)
    container_path = get_paths.get_container_path()
    command = ["sudo", "singularity", "build"]
    command += ["s2cnn_container.sif", "singularity_recipe"]
    print(f"running: {' '.join(command)}")
    subprocess.run(command, cwd=container_path)


def train_s2cnn(run_args, sub_args):
    path = Path(__file__).parent.joinpath("train_s2cnn.py").absolute()
    command = env_prefix(run_args.env) + ["python3", "-u", str(path)]
    command += sub_args
    print(f"running: {' '.join(command)}")
    subprocess.run(command)


def train_cnn(run_args, sub_args):
    path = Path(__file__).parent.joinpath("train_cnn.py").absolute()
    command = env_prefix(run_args.env) + ["python3", "-u", str(path)]
    command += sub_args
    print(f"running: {' '.join(command)}")
    subprocess.run(command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["local", "singularity"], default="local")

    sp = parser.add_subparsers(dest="subparser_name")

    build_sin_parser = sp.add_parser("build-singularity")
    build_sin_parser.set_defaults(func=build_sin)

    train_parser = sp.add_parser("train-s2cnn")
    train_parser.set_defaults(func=train_s2cnn)

    train_parser = sp.add_parser("train-cnn")
    train_parser.set_defaults(func=train_cnn)

    run_args, sub_args = parser.parse_known_args()

    run_args.func(run_args, sub_args)


if __name__ == "__main__":
    main()

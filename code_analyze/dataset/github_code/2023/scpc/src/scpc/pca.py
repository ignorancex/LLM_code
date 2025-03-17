# Copyright 2023 Sean Robertson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import argparse
import os

from typing import Optional, Sequence

import torch
import param

from pydrobert.param.serialization import register_serializer
from pydrobert.param.argparse import (
    add_serialization_group_to_parser,
    add_deserialization_group_to_parser,
)
from pydrobert.torch.data import SpectDataLoader, SpectDataLoaderParams

from .modules import Encoder

register_serializer("yaml")


class PCAParams(SpectDataLoaderParams):
    frames_per_batch = param.Integer(
        100,
        doc="Frames per batch element randomly drawn per batch to contribute to PCA. "
        "Unspecified means all frames",
    )


def _is_dir(val: str) -> str:
    if not os.path.isdir(val):
        raise ValueError(f"'{val}' is not a directory")
    return val


def _pos_int(val: str) -> int:
    val = int(val)
    if val <= 0:
        raise ValueError(f"{val} is not positive")
    return val


@torch.no_grad()
def main(args: Optional[Sequence[str]] = None):
    """Run PCA on SpectDataSet"""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("dir", type=_is_dir, help="SpectDataSet path")
    parser.add_argument(
        "ckpt", type=argparse.FileType("rb"), help="Path to pre-trained model"
    )
    parser.add_argument("dim", type=_pos_int, help="Dimension to reduce to")
    parser.add_argument("out", help="Where to write PCA V matrix to")
    parser.add_argument("--device", type=torch.device, default=None)
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Number of workers in datasets"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
    )

    add_serialization_group_to_parser(parser, PCAParams)
    add_deserialization_group_to_parser(parser, PCAParams, "params")

    options = parser.parse_args(args)

    if options.quiet:
        print_ = lambda x: None
    else:
        print_ = print

    if options.device is None:
        if torch.cuda.is_available():
            options.device = torch.device(torch.cuda.current_device())
        else:
            options.device = torch.device("cpu")

    print_("Loading model...")
    encoder = Encoder.from_checkpoint(options.ckpt).to(options.device)

    print_("Loading dataset...")
    dl = SpectDataLoader(
        options.dir,
        options.params,
        shuffle=False,
        num_workers=options.num_workers,
        pin_memory=options.device.type == "cuda",
    )

    if not options.quiet:
        try:
            from tqdm import tqdm

            dl = tqdm(dl)
        except ImportError:
            pass

    print_("Iterating through data to construct matrix...")
    A = []
    for feats, _, feat_sizes, _ in dl:
        feats, feat_sizes = feats.to(options.device), feat_sizes.to(options.device)
        reps, rep_sizes = encoder(feats, feat_sizes)
        del feats, feat_sizes
        reps = torch.nn.utils.rnn.pack_padded_sequence(
            reps, rep_sizes.cpu(), True, False
        ).data
        if options.params.frames_per_batch:
            idx = torch.randperm(reps.size(0), device=options.device)
            idx = idx[: options.params.frames_per_batch]
            reps = reps.index_select(0, idx).contiguous()
        A.append(reps)

    A = torch.cat(A)
    print_(f"matrix shape {A.shape}")

    # A = U diag(S) V^T
    # AV = U diag(S) V^TV
    # AV = U diag(S) (since V is unitary)
    # AV[..., :K] = U diag(S)[..., :K]
    print_(f"Performing PCA to dim {options.dim}")
    _, _, V = torch.pca_lowrank(A, options.dim)
    assert V.shape == (A.size(1), options.dim)
    if V.isnan().any():
        raise ValueError("PCA failed! matrix contains NaNs!")
    print_(f"Performed PCA. Saving...")
    torch.save(V.cpu(), options.out)
    print_(f"Saved")


if __name__ == "__main__":
    sys.exit(main())

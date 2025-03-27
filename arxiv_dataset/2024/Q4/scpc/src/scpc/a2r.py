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

import os
import argparse
import glob

from typing import Optional, Sequence
from itertools import chain

import torch
import numpy as np

from .expert import get_feature_extractor_parser, UpstreamExpert

import pydrobert.speech.util as sutil


def _is_dir(val: str) -> str:
    if not os.path.isdir(val):
        raise ValueError(f"'{val}' is not a directory")
    return val


@torch.no_grad()
def main(args: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Convert audio files to representation files",
        parents=[get_feature_extractor_parser()],
    )
    in_grp = parser.add_mutually_exclusive_group(required=True)
    in_grp.add_argument(
        "--in-dir",
        type=_is_dir,
        default=None,
        help="Path storing input audio files. Utt id is basename minus "
        "--audio-suffix. Output directory layout reproduces that of input",
    )
    in_grp.add_argument(
        "--in-map",
        type=argparse.FileType("r"),
        default=None,
        help="File containing <utt-id> <audio-file> pairs. Utterances stored flat "
        "in out_dir",
    )
    parser.add_argument(
        "ckpt", type=argparse.FileType("rb"), help="Path to pre-trained model"
    )
    parser.add_argument("out_dir", help="Path to store speech reps to")
    parser.add_argument(
        "--numpy",
        action="store_true",
        default=False,
        help="Whether to store as .npy files",
    )
    parser.add_argument(
        "--audio-suffix", default=".wav", help="Suffix of audio files (--in-dir only)"
    )
    parser.add_argument(
        "--device",
        default=None,
        type=torch.device,
        help="What device to compute features on. Defaults to CUDA if available",
    )
    parser.add_argument(
        "--force-as",
        default=None,
        choices={
            "table",
            "wav",
            "hdf5",
            "npy",
            "npz",
            "pt",
            "sph",
            "kaldi",
            "file",
            "soundfile",
        },
        help="Force audio files to be read as a specific type. table: kaldi table (key "
        "is utterance id); wav: wave file; hdf5: HDF5 archive (key is utterance id); "
        "npy: Numpy binary; npz: numpy archive (key is utterance id); pt: PyTorch "
        "binary; sph: NIST SPHERE file; kaldi: kaldi object; file: numpy.fromfile "
        "binary. soundfile: force soundfile processing.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=-1,
        help="Channel to draw audio from. Default is to assume mono",
    )

    options = parser.parse_args(args)

    if options.device is None:
        if torch.cuda.is_available():
            options.device = torch.device(torch.cuda.current_device())
        else:
            options.device = torch.device("cpu")

    expert = UpstreamExpert(options.ckpt, options=options).to(options.device)

    if options.in_dir is not None:
        in_dir = glob.escape(options.in_dir)
        it = chain(
            glob.iglob(f"{in_dir}/**/*{options.audio_suffix}"),
            glob.iglob(f"{in_dir}/*{options.audio_suffix}"),
        )
        it = (
            (
                os.path.basename(inf).rsplit(options.audio_suffix, 1)[0],
                inf,
                os.path.join(
                    options.out_dir,
                    os.path.relpath(os.path.dirname(inf), options.in_dir),
                ),
            )
            for inf in it
        )
    else:
        assert options.in_map is not None
        it = (
            str(x).strip().split(maxsplit=1) + [options.out_dir] for x in options.in_map
        )
    for utt_id, inf, odir in it:
        signal = sutil.read_signal(inf, dtype=np.float32, force_as=options.force_as)
        if options.channel == -1 and signal.ndim > 1 and signal.shape[0] > 1:
            raise ValueError(
                "Utterance {}: Channel is not specified but signal has "
                "shape {}".format(utt_id, signal.shape)
            )
        elif (options.channel != -1 and signal.ndim == 1) or (
            options.channel >= signal.shape[0]
        ):
            raise ValueError(
                "Utterance {}: Channel specified as {} but signal has "
                "shape {}".format(utt_id, options.channel, signal.shape)
            )
        if signal.ndim != 1:
            signal = signal[options.channel]
        signal = torch.from_numpy(signal).to(options.device)
        if signal.isnan().any():
            raise ValueError(f"'{inf}' contains NaN values!")
        assert not signal.isnan().any()
        reps = expert([signal])["hidden_states"][0].cpu()
        if reps.isnan().any():
            raise ValueError(f"representation of '{inf}' contains NaN values!")
        os.makedirs(odir, exist_ok=True)
        if options.numpy:
            np.save(os.path.join(odir, utt_id + ".npy"), reps.double().numpy())
        else:
            torch.save(reps, os.path.join(odir, utt_id + ".pt"))

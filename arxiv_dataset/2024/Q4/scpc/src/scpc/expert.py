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

import argparse
import json

from typing import Dict, List, Optional, Tuple

import torch

__all__ = [
    "get_feature_extractor_parser",
    "UpstreamExpert",
]


try:
    import pydrobert.speech.torch as storch
    import pydrobert.speech.pre as spre
    import pydrobert.speech.post as spost
    import pydrobert.speech.compute as scompute
    import pydrobert.speech.alias as salias
except ImportError:
    storch = spre = spost = scompute = salias = sutil = sconfig = None

from .modules import Encoder


def get_feature_extractor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@", add_help=False)
    parser.add_argument(
        "--preprocess",
        default="[]",
        help="JSON list of configurations for "
        "``pydrobert.speech.pre.PreProcessor`` objects. Audio will be "
        "preprocessed in the same order as the list",
    )
    parser.add_argument(
        "--postprocess",
        default="[]",
        help="JSON List of configurations for "
        "``pydrobert.speech.post.PostProcessor`` objects. Features will be "
        "postprocessed in the same order as the list",
    )
    feat_group = parser.add_mutually_exclusive_group(required=True)
    feat_group.add_argument(
        "--computer-json",
        default=None,
        help="Path to JSON configuration of a feature computer",
    )
    feat_group.add_argument(
        "--raw",
        action="store_true",
        default=False,
        help="If specified, audio will not be processed",
    )
    parser.add_argument(
        "--pca-file",
        default=None,
        type=argparse.FileType("rb"),
        help="Path to matrix for reducing dimensionality of representations",
    )
    return parser


class RawComputer(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1)


def _get_feature_extractor_from_opts(options) -> Tuple[torch.nn.Module, int]:
    if options.raw:
        computer, rate = RawComputer(), 1
    else:
        with open(options.computer_json) as f:
            json_ = json.load(f)
        computer = salias.alias_factory_subclass_from_arg(scompute.FrameComputer, json_)
        rate = computer.frame_shift
        if isinstance(computer, scompute.SIFrameComputer):
            computer = storch.PyTorchSIFrameComputer.from_si_frame_computer(computer)
        elif isinstance(computer, scompute.STFTFrameComputer):
            computer = storch.PyTorchSTFTFrameComputer.from_stft_frame_computer(
                computer
            )
        else:
            raise NotImplementedError(
                f"Unknown computer type {type(computer).__name__}"
            )
    computers = [computer]

    json_ = json.loads(options.preprocess)
    if len(json_):
        for json_i in json_[::-1]:
            preprocessor = salias.alias_factory_subclass_from_arg(
                spre.PreProcessor, json_i
            )
            if isinstance(preprocessor, spre.Dither):
                preprocessor = storch.PyTorchDither.from_dither(preprocessor)
            elif isinstance(preprocessor, spre.Preemphasize):
                preprocessor = storch.PyTorchPreemphasize.from_preemphasize(
                    preprocessor
                )
            else:
                raise NotImplementedError(
                    f"Unknown preprocessor type {type(preprocessor).__name__}"
                )
            computers.insert(0, preprocessor)

    json_ = json.loads(options.postprocess)
    if len(json_):
        computers.extend(
            storch.PyTorchPostProcessorWrapper.from_postprocessor(
                salias.alias_factory_subclass_from_arg(spost.PostProcessor, x)
            )
            for x in json_
        )

    if len(computers) > 1:
        computer = torch.nn.Sequential(computers)

    return computer, rate


class UpstreamExpert(torch.nn.Module):
    encoder: Encoder
    feat_extractor: torch.nn.Module
    pca: torch.nn.Module

    def __init__(
        self, ckpt: str, model_config: Optional[str] = None, options=None, **kwargs
    ):
        super().__init__()
        if model_config is not None:
            if options is not None:
                raise ValueError("Cannot specify both model_config and options")
            options = get_feature_extractor_parser().parse_args(["@" + model_config])
        elif options is None:
            raise ValueError("Either options or model_config must be specified")
        self.feat_extractor, self._feds_rate = _get_feature_extractor_from_opts(options)
        self.name = "[scpc]"
        self.encoder = Encoder.from_checkpoint(ckpt, "cpu")
        if self.encoder.input_size != 1 and options.raw:
            raise ValueError(
                "--raw specified but encoder expecting features of size "
                f"{self.encoder.input_size}"
            )
        if options.pca_file is not None:
            W = torch.load(options.pca_file, "cpu")
            if W.ndim != 2 or self.encoder.output_size != W.size(0):
                raise ValueError(
                    "Expected --pca-file to contain matrix of shape ("
                    f"{self.encoder.output_size}, ...); got {W.shape}"
                )
            self.pca = torch.nn.Linear(W.size(0), W.size(1))
            self.pca.weight.data.copy_(W.T)
            del W
        else:
            self.pca = torch.nn.Identity()

    def get_downsample_rates(self, key: str) -> int:
        rate = self.encoder.downsampling_factor
        if self.feat_extractor is not None:
            rate *= self._feds_rate
        return rate

    def forward(self, wavs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(wavs) == 0:
            return {"hidden_states": torch.empty(0, 0, self.encoder.output_size)}
        feats = [self.feat_extractor(x) for x in wavs]
        assert len(feats) == 1
        lens = torch.tensor([f.size(0) for f in feats]).to(wavs[0].device)
        x = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
        x, lens = self.encoder(x, lens)
        x = self.pca(x)
        return {"hidden_states": x}

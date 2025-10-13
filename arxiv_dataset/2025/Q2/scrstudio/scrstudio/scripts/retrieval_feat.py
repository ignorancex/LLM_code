from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Type

import cv2
import numpy as np
import torch
import tyro
from hloc import extractors
from hloc.extract_features import resize_image
from hloc.utils.base_model import dynamic_load
from nanopq import PQ
from tqdm import tqdm
from typing_extensions import Annotated

from scrstudio.configs.base_config import InstantiateConfig, PrintableConfig
from scrstudio.data.utils.readers import LMDBReaderConfig, ReaderConfig

device = torch.device("cuda")

@dataclass
class HLocDatasetConfig(InstantiateConfig):
    """Configuration for HLocDataset instantiation"""

    _target: Type = field(default_factory=lambda: HLocDataset)

    root: Path = Path("data/")

    reader: ReaderConfig = field(default_factory=LMDBReaderConfig)

    conf: dict = field(default_factory=dict)




class HLocDataset(torch.utils.data.Dataset):
    default_conf = {
        "globs": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        "grayscale": False,
        "resize_max": None,
        "resize_force": False,
        "interpolation": "cv2_area",  # pil_linear is more accurate but slower
        "size_multiple": 1,
    }

    def __init__(self, config: HLocDatasetConfig):
        self.conf = SimpleNamespace(**{**self.default_conf, **config.conf})
        self.root = config.root
        self.reader = config.reader.setup(root=self.root)

    def __getitem__(self, idx):
        image=self.reader[idx]
        if self.conf.grayscale and image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]

        if self.conf.resize_max and (
            self.conf.resize_force or max(size) > self.conf.resize_max
        ):
            scale = self.conf.resize_max / max(size)
            size_new = tuple(
                    int(round(x * scale / self.conf.size_multiple))
                    * self.conf.size_multiple
                    for x in size
                )
            image = resize_image(image, size_new, self.conf.interpolation)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.0

        data = {
            "image": image,
            "original_size": np.array(size),
        }
        return data

    def __len__(self):
        return len(self.reader)

@dataclass
class ComputeNetVLAD(PrintableConfig):
    """Download a dataset"""
    dataset: HLocDatasetConfig = field(default_factory=HLocDatasetConfig)
    seed: int = 42
    M: int = 256
    as_half: bool = False
    num_workers: int = 1
    data:Path = Path("data/")
    pq:bool = False

    def main(self) -> None:
        root = self.data
        dataset = HLocDatasetConfig(root=root, conf={"resize_max": 1024}).setup()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        Model = dynamic_load(extractors, "netvlad")
        model = Model({"name": "netvlad"}).eval().to(device)
        loader = torch.utils.data.DataLoader(
            dataset, num_workers=self.num_workers, shuffle=False, pin_memory=True
        )
        feats=[]
        with torch.no_grad():
            for idx, data in enumerate(tqdm(loader)):
                pred = model({"image": data["image"].to(device, non_blocking=True)})
                feat = pred['global_descriptor'][0].cpu().numpy()
                feats.append(feat)
                del pred

        feats = np.stack(feats)
        if self.pq:
            pq = PQ(M=self.M,verbose=False).fit(feats,seed=self.seed)
            codes=pq.encode(feats)
            with open(root / "netvlad_feats_pq.pkl", 'wb') as f:
                pickle.dump((pq, codes), f)

        if self.as_half:
            dt = feats.dtype
            if (dt == np.float32) and (dt != np.float16):
                feats = feats.astype(np.float16)
        np.save(root / "netvlad_feats.npy", feats)


def main(
    comp: ComputeNetVLAD
):

    comp.main()

Commands =    Annotated[ComputeNetVLAD, tyro.conf.subcommand(name="netvlad")]

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Commands))


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa



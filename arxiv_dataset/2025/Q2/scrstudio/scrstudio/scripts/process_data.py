from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import torch
import tyro

from scrstudio.configs.base_config import PrintableConfig
from scrstudio.data.utils.readers import folder2lmdb

device = torch.device("cuda")


@dataclass
class ConvertColmap(PrintableConfig):
    """Convert Colmap reconstructions to scrstudio format."""
    sfm_path: Path
    image_path: Path
    output_path: Path

    def main(self) -> None:
        rgb_path = self.output_path / "rgb"
        rgb_path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdirname:
            pycolmap.undistort_images(tmpdirname, str(self.sfm_path), str(self.image_path))
            
            recon=pycolmap.Reconstruction(f"{tmpdirname}/sparse")
            undistored_path=Path(tmpdirname) / "images"
            images=sorted([x for x in recon.images.values() if x.has_pose], key=lambda x: x.name)
            all_subdirs ={(rgb_path / image.name).parent for image in images }
            for subdir in all_subdirs:
                os.makedirs(subdir, exist_ok=True)
            calibs=[]
            names=[]
            poses=[]
            shapes=[]
            for image in images:
                pose=np.eye(4)
                pose[:3]=image.cam_from_world.inverse().matrix()
                poses.append(pose)
                intrinsics=np.eye(3)
                camera=recon.cameras[image.camera_id]
                assert camera.model== pycolmap.CameraModelId.PINHOLE
                intrinsics[[0, 1, 0, 1], [0, 1, 2, 2]]=camera.params
                calibs.append(intrinsics)
                shutil.copy(undistored_path / image.name, rgb_path / image.name)
                names.append(image.name)
                shapes.append((camera.height, camera.width))
            np.save(self.output_path / "calibration.npy", calibs)
            np.save(self.output_path / "poses.npy", poses)
            np.save(self.output_path / "image_shapes.npy", shapes)
            folder2lmdb(rgb_path, self.output_path / "rgb_lmdb")







def main(
    comp: ConvertColmap
):

    comp.main()

Commands =    ConvertColmap

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Commands))


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa



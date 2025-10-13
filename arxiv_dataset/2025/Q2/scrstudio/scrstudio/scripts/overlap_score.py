from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import torch
import tyro
from tqdm import tqdm
from typing_extensions import Annotated

from scrstudio.configs.base_config import PrintableConfig
from scrstudio.data.utils.readers import lmdb_image_shapes

device = torch.device("cuda")

class Frustum:
    def __init__(self,c2w,intrinsics,image_shapes):
        images_wh=image_shapes[:,1::-1]
        if intrinsics.ndim == 1: # M
            fxy=intrinsics[:,None]
            cxy = images_wh/2
        else: # M x 3 x 3
            fxy=intrinsics[:,[0,1],[0,1]]
            cxy=intrinsics[:,[0,1],[2,2]]
        self.min_xy=-cxy/fxy
        self.max_xy=(images_wh-cxy)/fxy
        self.c2w = c2w

    def __len__(self):
        return len(self.c2w)

@dataclass
class ComputeOverlap(PrintableConfig):
    """Download a dataset"""
    data: Path = Path("data/")
    seed: int = 0
    num_samples: int = 1024
    min_depth: float = 0.1
    max_depth: float = 8
    def main(self) -> None:
        generator = torch.Generator(device=device).manual_seed(self.seed)
        poses=self.data/'poses.npy'
        intrinsics=self.data/'calibration.npy'
        image_shapes=self.data/'image_shapes.npy'
        if not image_shapes.exists():
            lmdb_image_shapes(self.data)
        c2w=np.load(poses) # N x 4 x 4 (camera to world)
        intrinsics=np.load(intrinsics) # N x 3 x 3
        image_shapes=np.load(image_shapes) # N x 3 (h,w, c )
        frustum= Frustum(c2w,intrinsics,image_shapes)
        src_frustum=dst_frustum=frustum
        pose_covis = torch.zeros((len(src_frustum), len(dst_frustum)),device=device)
        c2w=torch.tensor(src_frustum.c2w,dtype=torch.float32).cuda()
        w2c=torch.inverse(torch.tensor(dst_frustum.c2w,dtype=torch.float32).cuda())
        depth_samples=torch.empty(self.num_samples,dtype=torch.float32).cuda().uniform_(self.min_depth,self.max_depth,generator=generator)
        dst_frustum_min_xy=torch.tensor(dst_frustum.min_xy,dtype=torch.float32).cuda()
        dst_frustum_max_xy=torch.tensor(dst_frustum.max_xy,dtype=torch.float32).cuda()
        for i in tqdm(range(len(src_frustum))):
            w_samples=torch.empty(self.num_samples,dtype=torch.float32).cuda().uniform_(src_frustum.min_xy[i,0],src_frustum.max_xy[i,0],generator=generator)
            h_samples=torch.empty(self.num_samples,dtype=torch.float32).cuda().uniform_(src_frustum.min_xy[i,1],src_frustum.max_xy[i,1],generator=generator)
            src_coords=torch.stack([w_samples,h_samples,depth_samples],dim=1)
            reproj_mat = w2c @ c2w[i]
            src_dir= reproj_mat[:,:3,:3] @ src_coords.T
            dst_coords = src_dir + reproj_mat[:,:3,3:4]
            dst_depths = dst_coords[:,2]
            dst_xy = dst_coords[:,:2,:]/dst_coords[:,2:]
            score=(torch.cosine_similarity(src_dir,dst_coords,dim=1)+1)*0.5
            mask =(dst_depths>self.min_depth) & (dst_depths<self.max_depth)
            mask &= (dst_xy >= dst_frustum_min_xy[...,None]).all(dim=1) & (dst_xy < dst_frustum_max_xy[...,None]).all(dim=1) 
            mask=score*mask
            pose_covis[i]=mask.sum(dim=1)/self.num_samples
        pose_covis = pose_covis.cpu().numpy()
        np.fill_diagonal(pose_covis,0)
        pose_covis_sym=2*pose_covis*pose_covis.T/(pose_covis+pose_covis.T)
        np.fill_diagonal(pose_covis_sym,0)
        pose_covis_sym=np.nan_to_num(pose_covis_sym)
        coo_covis=scipy.sparse.coo_array(pose_covis_sym)
        scipy.sparse.save_npz(self.data/'pose_overlap.npz',coo_covis)



def main(
    comp: ComputeOverlap
):

    comp.main()

Commands =    Annotated[ComputeOverlap, tyro.conf.subcommand(name="netvlad")]

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Commands))


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa



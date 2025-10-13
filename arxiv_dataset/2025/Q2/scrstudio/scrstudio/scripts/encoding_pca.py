from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
import tyro
from cuml import PCA
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Annotated

from scrstudio.configs.base_config import PrintableConfig
from scrstudio.data.datasets.camloc_dataset import CamLocDatasetConfig
from scrstudio.encoders.base_encoder import EncoderConfig
from scrstudio.encoders.dedode_encoder import DedodeEncoderConfig
from scrstudio.encoders.loftr_encoder import LoFTREncoderConfig

device = torch.device("cuda")

def get_prefix(encoder):
    if isinstance(encoder, LoFTREncoderConfig):
        return encoder.model
    elif isinstance(encoder, DedodeEncoderConfig):
        return f"d3{encoder.detector}{encoder.descriptor}"
    else:
        return encoder.__class__.__name__.replace("EncoderConfig", "").lower()

def pca_to_state_dict(pca):
    torch_mean=torch.as_tensor(pca.mean_, device='cuda')
    torch_components=torch.as_tensor(pca.components_, device='cuda')
    torch_bias= -torch_mean@torch_components.T

    torch.as_tensor(pca.mean_, device='cuda')
    return {'weight':torch_components[:,:,None,None],'bias':torch_bias}
@dataclass
class ComputePCA(PrintableConfig):
    """Download a dataset"""

    n_components:int = 128
    max_samples:int = 2000
    memory_ratio:float = 1
    prefix: str = ""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    dataset: CamLocDatasetConfig = field(default_factory=CamLocDatasetConfig)

    data: Path = Path("data/")
    output: Optional[Path] = None
    save_pkl:bool = False
    num_workers:int = 4

    def get_max_samples(self):
        return self.max_samples

    def get_features(self):
        print(self.encoder)
        encoder=self.encoder.setup()
        encoder = encoder.to(device)
        encoder.eval()

        self.dataset.data=self.data
        ds=self.dataset.setup(preprocess=encoder.preprocess)
        dl= DataLoader(ds,shuffle=False, batch_size=1,
                       num_workers=self.num_workers, pin_memory=True)
        total_memory=torch.cuda.get_device_properties(device).total_memory
        self.out_channels=encoder.out_channels
        print(f"GPU memory total: {total_memory/1e9:.2f} GB")

        samples = int(total_memory *0.48 / (encoder.out_channels * 4 * len(dl)) * self.memory_ratio)
        samples = min(samples, self.get_max_samples())

        all_feats=torch.empty(samples*len(dl),encoder.out_channels,dtype=torch.float32,device=device)
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")
        print(f"all_feats memory allocated: {all_feats.numel()*4/1e9:.2f} GB with {samples} samples")

        for i,batch in enumerate(tqdm(dl)):
            image = batch['image'].to(device, non_blocking=True)
            with torch.inference_mode():
                with autocast("cuda",enabled=True):
                    output=encoder.keypoint_features({"image": image},n=samples)
                    all_feats[i*samples:(i+1)*samples]=output["descriptors"]
        return all_feats

    def main(self) -> None:
        """Download the dataset"""
        prefix = self.prefix if self.prefix else get_prefix(self.encoder)
        proc_dir=self.data / "proc"
        proc_dir.mkdir(parents=True, exist_ok=True)
        all_feats=self.get_features()
        torch.cuda.empty_cache()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")
        pca_torch = PCA(n_components = self.n_components,copy=False)
        pca_torch.fit(all_feats)

        print(f"Explained variance: {pca_torch.explained_variance_ratio_.sum()}")

        output = proc_dir/  f"pca{prefix}_{self.n_components}.pkl" if self.output is None else self.output
        if self.save_pkl:
            with open(output, "wb") as f:
                pickle.dump(pca_torch, f)
            print(f"Saved PCA to {output}")

        state_dict=pca_to_state_dict(pca_torch)


        # test
        input_rand=torch.rand(1024,self.out_channels,device=device)
        gt_output=torch.as_tensor(pca_torch.transform(input_rand),device=device)
        conv=torch.nn.Conv2d(self.out_channels,self.n_components,1,dtype=torch.float32).to(device)
        conv.load_state_dict(state_dict)
        conv.eval()
        with torch.no_grad():
            input_reshaped=input_rand.view(-1,self.out_channels,1,1)
            print(f"Max diff: {torch.max(torch.abs(conv(input_reshaped).view(-1,self.n_components)-gt_output))}")

        path=output.with_suffix(".pth")
        torch.save(state_dict, path)
        print(f"Saved state dict to {path}")

@dataclass
class LoFTRPCA(ComputePCA):
    encoder: LoFTREncoderConfig = field(default_factory=LoFTREncoderConfig)



@dataclass
class DedodePCA(ComputePCA):
    encoder: DedodeEncoderConfig = field(default_factory=DedodeEncoderConfig)
    def get_max_samples(self):
        return self.encoder.k

    
def main(
    comp: ComputePCA
):

    comp.main()

Commands = Union[
    Annotated[LoFTRPCA, tyro.conf.subcommand(name="loftr")],
    Annotated[DedodePCA, tyro.conf.subcommand(name="dedode")],
]

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Commands))


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa



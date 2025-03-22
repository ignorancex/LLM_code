import argparse
import sys
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')
sys.path.append(mammoth_path + '/utils')

from datasets.seq_nturgbd60 import Ntu60
from datasets.transforms.ntu_transforms import ntu_to_tensor

from utils.conf import get_device


class Interpolator(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.interval = args.sampling_interval

    @torch.no_grad()
    def forward(self, sample, indexes):
        return F.interpolate(sample,
                             size=(120, sample.shape[3], sample.shape[4]),
                             mode='trilinear',
                             align_corners=False)

    def get_rebuilding_factor(self):
        return 1 / self.interval

    def get_efficiency_ratio(self):
        return 1 - self.get_rebuilding_factor()

    def get_efficient_buffer_sample(self, not_aug_inputs: torch.Tensor) -> Tuple[torch.Tensor, None]:

        not_aug_inputs = not_aug_inputs[:, :, ::self.interval, :, :]

        return not_aug_inputs, torch.arange(0, not_aug_inputs.shape[2], self.interval).unsqueeze(0).repeat(not_aug_inputs.shape[0], 1)


def test_interpolator(args):
    dataset = Ntu60(args.data_path, split="train", p_interval=[0.5, 1], window_size=120, transform=transforms.Compose([
        ntu_to_tensor(),
    ]), permute=1)
    dataset = torch.from_numpy(dataset.data)

    for interval in range(1, 21):
        args.sampling_interval = interval
        print(f"Testing with temporal ratio: {interval}...")

        interpolator = Interpolator(args, get_device())

        print("Sampling uniformly the dataset...")
        uniformly_sampled, _ = interpolator.get_efficient_buffer_sample(dataset.data)

        print("Reconstructing the uniformly sampled data...")
        reconstructed = interpolator(uniformly_sampled, None)

        print("Computing the reconstruction error...")
        reconstruction_error = torch.nn.functional.mse_loss(reconstructed, dataset.data)

        print(reconstruction_error.item())
        with open("reconstruction_error.txt", "a") as f:
            f.write(f"A frame taken every {interval}, efficiency ratio of {interpolator.get_efficiency_ratio():.2%} ---> Reconstruction error: {reconstruction_error.item()}\n")


if __name__ == '__main__':
    print("""
      @@@@
     @   @@
    @@   @ @       @@@@@
   @ @   @  @@@@@ @  @  @
   @ @   @   @   @   @   @@
  @  @   @   @   @   @   @ @
  @  @   @   @   @   @   @  @
 @   @   @   @   @   @   @   @
@-----------------------------@-------------------@
     <--->                     @ @   @   @   @   @
       T                        @@   @   @   @  @
                                 @   @   @   @ @
                                  @  @   @   @ @
                                   @ @   @   @@
                                    @@   @   @@
                                     @   @   @
                                      @@ @  @
                                       @ @ @
                                        @@@
""")

    parser = argparse.ArgumentParser(description='Test the recostruction error of the linear interpolator.')

    parser.add_argument('--data_path', type=str, default="./data/NTU60_XView.npz", help='Path to the data dir')
    # parser.add_argument("--rebuilder", type=str, default="interpolator", choices=get_all_rebuilders(), help="Select rebuilder to test.")
    parser.add_argument('--sampling_interval', type=int, default=4, help='Interval between frames.')

    args = parser.parse_args()

    test_interpolator(args)

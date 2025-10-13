from typing import Union, List, Tuple, Type

from torch import nn
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm


class DifferenceWeightingBlock(nn.Module):
    def __init__(self, features_per_stage: Union[int, List[int], Tuple[int, ...]], conv_op: Type[_ConvNd]):
        super().__init__()

        for norm in range(len(features_per_stage)):
            setattr(self, f"norm_{norm}", get_matching_instancenorm(conv_op)(features_per_stage[norm], affine=True))
    
    def forward(self, skips_current, skips_prior,skips_weights):
        #print("------- DIFF WEIGHTING BLOCK ----------")
        for num_skip, (skip_current, skip_prior,skip_weights) in enumerate(zip(skips_current, skips_prior,skips_weights)):
            #print("num skip ",num_skip)
            #print(skip_prior.shape,skip_current.shape,skip_weights)
            weighting = skip_weights[0]*skip_current - skip_weights[1]*skip_prior

            norm = getattr(self, f"norm_{num_skip}")
            weighting = norm(weighting)

            skips_current[num_skip] = skip_current * weighting + skip_current
        return skips_current
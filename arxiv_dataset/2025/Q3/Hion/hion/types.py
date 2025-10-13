from typing import Callable, TypedDict
import torch as th

"""
Author: Josue N Rivera
"""

LagrangianFunc = Callable[[th.DoubleTensor, th.DoubleTensor, th.DoubleTensor], th.DoubleTensor]

TrainingConfigDict = TypedDict('TrainingConfigDict', {'number of epoch': int, 
 'optimizer': dict,
 'losses': dict
 })

DistributionConfigDict = TypedDict('DistributionConfigDict',{
    'time': dict,
    'state': dict,
    'sample': dict
})

ControllerConfigDict = TypedDict('ControllerConfigDict', {
    'name': str,
    'args': dict
})

class ConfigDict(TypedDict):
    name: str
    training: TrainingConfigDict
    distribution: DistributionConfigDict
    controller: ControllerConfigDict
    dynamics: dict
    lagrangian: dict
    checkpoint: dict

class CheckpointDcit(TypedDict):
    name: str
    checkpoint: dict



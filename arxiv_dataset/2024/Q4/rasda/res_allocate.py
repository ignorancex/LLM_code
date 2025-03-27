from typing import Dict, Any
import logging
import numpy as np

from ray.tune.experiment import Trial
from ray.tune.execution.placement_groups import PlacementGroupFactory

logger = logging.getLogger(__name__)

def create_resource_allocation(base_cpus, base_gpus, num_workers, scaling_factor, min_t, max_t):
    def resource_allocation_function(tune_controller: "TuneController",
                trial: Trial,
                result: Dict[str, Any],
                scheduler: "ResourceChangingScheduler"):
        return successive_resource_doubling(
            base_cpus=base_cpus,
            base_gpus=base_gpus,
            num_workers=num_workers,
            scaling_factor=scaling_factor,
            min_t=min_t,
            max_t=max_t,
            trial=trial,
            result=result
        )
    return resource_allocation_function

def successive_resource_doubling(
    base_cpus: int,
    base_gpus: int,
    num_workers: int,
    scaling_factor: int,
    min_t: int,
    max_t: int,
    trial: Trial,
    result: Dict[str, Any]
    ):
        
    # Calculate the number of rungs and the rung milestones following the geometric progression as performed in ASHA
    num_rungs = np.floor(np.log(max_t/min_t) / np.log(scaling_factor))
    milestones = min_t * scaling_factor ** np.arange(int(num_rungs)+1)
    milestones = milestones.tolist()
    
    # If a milestone is reached, perform a resource doubling step by adjusting the bundles, otherwise do not change anything
    if result["training_iteration"] in milestones:
        increase_factor = scaling_factor**(milestones.index(result["training_iteration"]) + 1)
        new_bundles = [{"CPU": base_cpus, "GPU": base_gpus}] * int(num_workers) * increase_factor
        
        logger.warning(f"New bundles of Trial {trial} are {new_bundles}")    
        pgf = PlacementGroupFactory(new_bundles)
        pgf._head_bundle_is_empty = True
        
        return pgf
    else:
        return None
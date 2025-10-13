import os
from typing import Optional
from typing import Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.evaluator.metric import _to_cpu

from offsetocc.registry import METRICS


@METRICS.register_module()
class DumpOccResults(BaseMetric):
    """Dump model predictions to a pickle file for offline evaluation.

    Args:
        out_dir_path (str): Path of the dumped file. Must end with '.pkl'
            or '.pickle'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        collect_dir: (str, optional): Synchronize directory for collecting data
            from different ranks. This argument should only be configured when
            ``collect_device`` is 'cpu'. Defaults to None.
            `New in version 0.7.3.`
    """

    def __init__(self,
                 out_dir_path: str,
                 collect_device: str = 'cpu',
                 collect_dir: Optional[str] = None) -> None:
        super().__init__(
            collect_device=collect_device, collect_dir=collect_dir)
        self.out_dir_path = out_dir_path

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """transfer tensors in predictions to CPU."""
        data_samples = _to_cpu(data_samples)
        for data_sample in data_samples:
            pred_occ_map = data_sample['pred_occ_map']['occ_map']
            os.makedirs(os.path.join(self.out_dir_path, data_sample['token']), exist_ok=True)
            np.savez(os.path.join(self.out_dir_path, data_sample['token'], 'predictions'), semantics=pred_occ_map[0])

            # # remove anything that is not data_sample['pred_occ_map']
            # for key in list(data_sample.keys()):
            #     if key != 'pred_occ_map':
            #         data_sample.pop(key)

    def compute_metrics(self, results: list) -> dict:
        return {}
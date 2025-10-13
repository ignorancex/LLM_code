import torch
from torch import Tensor
from operator import itemgetter
from typing import Union, Dict, Literal


class Normalizer:

    def __init__(self) -> None:
        super().__init__()
        self.norm_stats = dict()   
        
    @torch.no_grad()
    def fit(
        self,
        data: Union[Dict, Tensor],
        mode: Union[Dict, str]
    ) -> None:
        """
        Fit a normalizer on @data.

        Args:
            data (dict or torch.Tensor): data used for fitting normalizers.
                Note that tensor shapes must be (N, D), where N is the
                number of elements and D is data dimension.
            mode (dict or str): normalization mode(s).
                - 'mean_std': x = (x - x_mean) / x_std
                - 'max_min': x = -1 + 2 * (x - x_min) / (x_max - x_min)
        """
        def _get_norm_stats(data: Tensor, mode: Literal['max_min', 'mean_std']) -> Dict:    
            assert len(data.shape) == 2
            assert mode in ['max_min', 'mean_std']
            if mode == 'max_min':
                eps = 1e-8
                data_max = data.max(dim=0)[0] - eps
                data_min = data.min(dim=0)[0] + eps
                norm_stats = {
                    'mode': mode,
                    'stats': {
                        'max': data_max,
                        'min': data_min
                    }
                }
            elif mode == 'mean_std':
                data_mean = data.mean(dim=0)
                data_std = data.std(dim=0)
                data_std = torch.clip(data_std, 1e-2, torch.inf)
                norm_stats = {
                    'mode': mode,
                    'stats': {
                        'mean': data_mean,
                        'std': data_std
                    } 
                }
            return norm_stats
        
        if isinstance(data, Tensor):
            norm_stats = _get_norm_stats(data, mode)
            self.norm_stats = norm_stats
        elif isinstance(data, dict):
            self.norm_stats.fromkeys(data.keys())
            for k, v in data.items():
                norm_stats = _get_norm_stats(v, mode[k])
                self.norm_stats[k] = norm_stats     
        
    def normalize(
        self,
        inputs: Union[Dict, Tensor],
        key: Union[str, None] = None
    ) -> Union[Dict, Tensor]:
        """
        Args:
            key (str or None, optional): if given, return normalized @inputs associated with @key.
        """
        def _normalize(
            input: Tensor,
            norm_stats: Dict
        ) -> Tensor:   
            if norm_stats['mode'] == 'mean_std':
                stats_keys = ['mean', 'std']
                mean, std = itemgetter(*stats_keys)(norm_stats['stats'])
                assert input.shape[-1] == mean.shape[-1]
                mean, std = mean.to(input.device), std.to(input.device)
                input = (input - mean) / std
            elif norm_stats['mode'] == 'max_min':
                stats_keys = ['max', 'min']
                max, min = itemgetter(*stats_keys)(norm_stats['stats'])
                assert input.shape[-1] == max.shape[-1]
                max, min = max.to(input.device), min.to(input.device)
                input = -1 + 2 * (input - min) / (max - min)
            return input
        
        if key is not None:
            assert key in self.norm_stats.keys()
        if isinstance(inputs, Tensor):
            norm_stats = self.norm_stats if key is None else self.norm_stats[key]
            inputs = _normalize(inputs, norm_stats)
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                if isinstance(v, dict):
                    inputs[k] = self.normalize(v)
                elif isinstance(v, Tensor):
                    if k in self.norm_stats.keys():
                        norm_stats = self.norm_stats[k]
                        inputs[k] = _normalize(v, norm_stats)
        return inputs

    def unnormalize(
        self,
        inputs: Union[Dict, Tensor],
        key: Union[str, None] = None
    ) -> Union[Dict, Tensor]:
        
        def _unnormalize(
            input: Tensor,
            norm_stats: Dict
        ) -> Tensor:  
            if norm_stats['mode'] == 'mean_std':
                stats_keys = ['mean', 'std']
                mean, std = itemgetter(*stats_keys)(norm_stats['stats'])
                assert input.shape[-1] == mean.shape[-1]
                mean, std = mean.to(input.device), std.to(input.device)
                input = input * std + mean
            elif norm_stats['mode'] == 'max_min':
                stats_keys = ['max', 'min']
                max, min = itemgetter(*stats_keys)(norm_stats['stats'])
                assert input.shape[-1] == max.shape[-1]
                max, min = max.to(input.device), min.to(input.device)
                input = min + (input + 1) * (max - min) / 2
            return input
        
        if key is not None:
            assert key in self.norm_stats.keys()
        if isinstance(inputs, Tensor):
            norm_stats = self.norm_stats if key is None else self.norm_stats[key]
            inputs = _unnormalize(inputs, norm_stats)
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                if isinstance(v, dict):
                    inputs[k] = self.normalize(v)
                elif isinstance(v, Tensor):
                    if k in self.norm_stats.keys():
                        norm_stats = self.norm_stats[k]
                        inputs[k] = _unnormalize(v, norm_stats)
        return inputs
    
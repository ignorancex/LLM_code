# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


from functools import partial
from typing import Any, Callable, Dict, List, Optional

from torch import nn


class EvaluationHook:
    """
    Base class for all evaluation hooks.
    """

    metric_dims: Dict[str, int] = {}

    def __init__(self):
        self._batch_output = {}
        self._handles = []

    def collect_results(self, module, inputs, kwargs, outputs):
        raise NotImplementedError()

    def attach_to(self, model: nn.Module):
        raise NotImplementedError()

    def __call__(self, module, inputs, kwargs, outputs, attached_to: Optional[str]):
        output = self.collect_results(module, inputs, kwargs, outputs)
        if attached_to is None:
            attached_to = "."
        for quantity, value in output.items():
            assert (
                value.shape[-1] == self.metric_dims[quantity]
            ), f"{quantity}: {value.shape} != {self.metric_dims[quantity]}"

        output = {attached_to: output}

        self._batch_output.update(output)

    def _attach_to(self, module: nn.Module, attached_to: Optional[str] = None):
        hook_with_name = partial(self, attached_to=attached_to)

        self._handles.append(module.register_forward_hook(hook_with_name, with_kwargs=True))

    def remove(self):
        for handle in self._handles:
            handle.remove()

    def finalize(self, stats: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return stats

    def _post_process_batch(
        self, batch_stats: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        return batch_stats

    @property
    def batch_output(self):
        batch_output = self._post_process_batch(self._batch_output)
        return batch_output

    def reset(self):
        self._batch_output = {}


# Utilities to merge dictionaries with overlapping keys
def merge_dicts(orig_dict: Dict[str, Any], add_dict: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in add_dict.items():
        if k in orig_dict:
            orig_dict[k] = merge_dicts(orig_dict[k], v)
        else:
            orig_dict[k] = v

    return orig_dict


# Utility to collect the internal state of all evaluation hooks
class CollectHooksOutput:
    def __init__(
        self,
        model: Callable,
        hooks: List[EvaluationHook],
        preprocess_batch: Optional[Callable],
    ):
        self.hooks = hooks
        self.model = model
        self.preprocess_batch = preprocess_batch

    def __call__(self, batch: Any):
        if self.preprocess_batch is not None:
            batch = self.preprocess_batch(batch)

        if isinstance(batch, dict):
            self.model(**batch)
        else:
            self.model(batch)

        outputs: Dict[str, Any] = {}
        for hook in self.hooks:
            outputs = merge_dicts(outputs, hook.batch_output)
            hook.reset()

        return outputs

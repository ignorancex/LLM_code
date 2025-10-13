from collections.abc import Sequence

import torch


class RAMCache(object):

    def __init__(self, dataset: dict[str, tuple]):
        # prevents infinite recursion from self.data = {'a': 'v1', 'b': 'v2'}
        # as now we have __setattr__, which will call __getattr__ when the line
        # self.data[k] tries to access self.data, won't find it in the instance
        # dictionary and return self.data[k] will in turn call __getattr__
        # for the same reason and so on.... so we manually set data initially
        super(RAMCache, self).__setattr__('dataset', dataset)
        super(RAMCache, self).__setattr__('cache', {})

    def __setattr__(self, k, v):
        if k == 'transform':
            self.clear_cache()
        elif k in ('dataset', 'cache'):
            super(RAMCache, self).__setattr__(k, v)
        setattr(self.dataset, k, v)

    def __getattr__(self, k):
        # we don't need a special call to super here because getattr is only
        # called when an attribute is NOT found in the instance's dictionary
        if k == 'cache':
            return self.cache
        if k == 'clear_cache':
            return self.clear_cache
        if k == 'dataset':
            return self.dataset
        return getattr(self.dataset, k)

    def __getitem__(self, k):
        if k in self.cache:
            return self.cache[k]
        else:
            item = self.dataset[k]
            self.cache[k] = self._to_cpu(item)
            return item

    def __len__(self):
        return len(self.dataset)

    def _to_cpu(self, item):
        """Recursively move all tensors found to CPU."""
        if isinstance(item, torch.Tensor):
            return item.cpu()
        if isinstance(item, (tuple, list, Sequence)):
            return [self._to_cpu(i) for i in item]
        # if isinstance(item, PIL.Image.Image):
        #     return item
        return item

    def clear_cache(self):
        super(RAMCache, self).__setattr__('cache', {})

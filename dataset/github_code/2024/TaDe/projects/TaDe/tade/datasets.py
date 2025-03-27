from mmengine.registry import DATASETS
from mmengine.dataset import BaseDataset


@DATASETS.register_module()
class BEVDataset(BaseDataset):
    pass

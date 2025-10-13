from dataclasses import dataclass
from typing import Callable

from .torch_wrapper import get_dataset_and_collate_fn_from_csv

from . import landscape
from . import greatesthits

DATASETS = {
    "landscape": landscape,
    "greatesthits": greatesthits
}

@dataclass
class DatasetInfo(object):
    dataset: object
    collate_fn: Callable
    size: int


def get_dataset_info(dataset_names, dataset_types, **kwargs) -> DatasetInfo:
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    if isinstance(dataset_types, str):
        dataset_types = [dataset_types]
    
    assert isinstance(dataset_names, (list, tuple)) 
    assert isinstance(dataset_types, (list, tuple))
    assert len(dataset_names) == len(dataset_types)

    # currently only support one dataset
    assert len(dataset_names) == 1

    csv_list = []
    for dataset_name, dataset_type in zip(dataset_names, dataset_types):
        mod = DATASETS[dataset_name]
        # check specified dataset_type is surely supported by the dataset
        dataset_type = dataset_type.lower()
        assert dataset_type in mod._METADATA_URLS, \
            f"dataset_type for dataset '{dataset_name}' must be one of {list(mod._METADATA_URLS.keys())}" 

        csv_list.append(mod._METADATA_URLS[dataset_type])

    dataset, collate_fn = get_dataset_and_collate_fn_from_csv(csv_list[0], **kwargs)

    return DatasetInfo(dataset, collate_fn, len(dataset))
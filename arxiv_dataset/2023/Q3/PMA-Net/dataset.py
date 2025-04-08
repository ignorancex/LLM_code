import json
import random
from typing import Union, List, Callable, Optional

from braceexpand import braceexpand
from transformers.utils import logging
import webdataset as wds


logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


def create_dataset(required_datasets: Union[str, List[str]], map_fn: Optional[Callable] = None, batch_size=8,
                   shuffle=False, repeat=False, dataset_rate=1.0, dataset_machine_rate=0.0,
                   *args, **kwargs) -> wds.DataPipeline:
    datasets_file = 'configs/datasets/datasets.json'
    with open(datasets_file, 'r') as f:
        available_datasets = json.load(f)
    available_dataset_names = list(available_datasets.keys())
    available_datasets = {name: braceexpand(path) for name, path in available_datasets.items()
                          if name in required_datasets}
    if not available_datasets:
        raise ValueError(f'{required_datasets} not in {available_dataset_names}')

    logger.info(f'Loading {required_datasets}...')

    shards = [shard for dataset in list(available_datasets.values()) for shard in dataset]

    shards = shards[:int(round(len(shards)*dataset_rate))]
    if dataset_machine_rate > 0:
        coco_shards = [s for s in shards if 'coco' in s]
        machine_shards = [s for s in shards if 'coco' not in s]
        machine_shards = machine_shards[:int(round(len(coco_shards))*dataset_machine_rate)]
        shards = coco_shards + machine_shards
        random.shuffle(shards)

    ds = wds.DataPipeline(
        wds.ResampledShards(shards) if repeat else wds.SimpleShardList(shards),
        wds.split_by_worker,
        wds.split_by_node,
        wds.tarfile_to_samples(),
        wds.shuffle(1000) if shuffle else None,
        wds.decode('pil'),
        wds.map(map_fn) if map_fn else None,
        wds.batched(batch_size),
        *args,
        **kwargs
    )
    return ds

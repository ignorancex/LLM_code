"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .cityscapes import CitySegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation

datasets = {
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'citys': CitySegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)

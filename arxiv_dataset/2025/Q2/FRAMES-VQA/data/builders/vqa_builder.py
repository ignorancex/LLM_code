from data.builders.base_dataset_builder import BaseDatasetBuilder

from common.registry import registry
from data.datasets.coco_vqa import *
from data.datasets.gqa_datasets import *
from pathlib import Path
import warnings
import os


@registry.register_builder("coco_vqa_raw")
class COCOVQABuilder_Raw(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/defaults_vqa_raw.yaml",
        "eval": "configs/datasets/vqa/eval_vqa_raw.yaml",
    }

@registry.register_builder("gqa_raw")
class GQABuilder_Raw(BaseDatasetBuilder):
    train_dataset_cls = GQADataset_Raw
    eval_dataset_cls = GQAEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa/defaults_gqa_raw.yaml",
        "eval": "configs/datasets/gqa/defaults_gqa_raw.yaml"
    }

@registry.register_builder("gqa_ood")
class GQAOODBuilder_Raw(BaseDatasetBuilder):
    train_dataset_cls = GQADataset_Raw
    eval_dataset_cls = GQAEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa/gqa-ood.yaml",
        "eval": "configs/datasets/gqa/gqa-ood.yaml"
    }


@registry.register_builder("coco_vqa_cp")
class COCOVQACPBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/vqa_cp.yaml"
    }


@registry.register_builder("coco_vqa_rephrasings")
class COCOVQA_Rephrasings_Builder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/vqa_rephrasings.yaml"
    }

@registry.register_builder("coco_vqa_ce")
class COCOVQACEBuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "configs/datasets/vqa/vqa_ce.yaml"
    }

@registry.register_builder("coco_cv-vqa")
class COCOCVVQABuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "configs/datasets/vqa/cv-vqa.yaml"
    }

@registry.register_builder("coco_iv-vqa")
class COCOIVVQABuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "configs/datasets/vqa/iv-vqa.yaml"
    }

@registry.register_builder("coco_advqa")
class COCOADVQABuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "configs/datasets/vqa/advqa.yaml"
    }

@registry.register_builder("textvqa")
class TextVQABuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "configs/datasets/vqa/textvqa.yaml"
    }

@registry.register_builder("vizwiz")
class VizWizBuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "configs/datasets/vqa/vizwiz.yaml"
    }

@registry.register_builder("coco_okvqa")
class COCOOKVQABuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "configs/datasets/vqa/ok-vqa.yaml"
    }
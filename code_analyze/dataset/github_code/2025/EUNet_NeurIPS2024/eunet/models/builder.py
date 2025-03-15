from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
ACCURACY = MODELS
SIMULATORS = MODELS
PREPROCESSOR = MODELS

def build_preprocessor(cfg):
    return PREPROCESSOR.build(cfg)


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)


def build_simulator(cfg):
    return SIMULATORS.build(cfg)

def build_accuracy(cfg):
    return ACCURACY.build(cfg)

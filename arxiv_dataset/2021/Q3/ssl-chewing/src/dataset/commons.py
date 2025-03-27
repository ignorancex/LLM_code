from enum import Enum


class SampleType(Enum):
    WINDOW = 0
    CHEW = 1
    BOUT = 2
    MEAL = 3


class SubsetType(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


class PartitionMode(Enum):
    LOSO_SIMPLE = 1  # 'leave one subject out, simple random split for train-validation sets'
    LOFTO_SIMPLE = 2  # 'leave one food type out, simple random split for train-validation sets'


class GroundTruthType(Enum):
    CHEW = 1
    BOUT = 2
    MEAL = 3

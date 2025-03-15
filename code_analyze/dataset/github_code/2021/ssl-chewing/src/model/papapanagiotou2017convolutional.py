"""
Models from the paper: https://ieeexplore.ieee.org/document/8037060/
"""

from typing import List

from tensorflow.keras.activations import relu, sigmoid, softmax
from tensorflow.keras.layers import Layer, Conv1D, MaxPool1D, Dense, Dropout, Flatten

from utilities.typingutils import is_typed_list


def _is_list_of_5_ints(x) -> bool:
    return is_typed_list(x, int) and len(x) == 5


def _feature_extraction_layers(filters: List[int], kernel_size: List[int], pool_size: List[int]) -> List[Layer]:
    assert _is_list_of_5_ints(filters)
    assert _is_list_of_5_ints(kernel_size)
    assert _is_list_of_5_ints(pool_size)

    lst: List[Layer] = []
    for i in range(5):
        lst.append(Conv1D(filters[i], kernel_size[i], activation=relu))
        lst.append(MaxPool1D(pool_size[i]))
    lst.append(Flatten())

    return lst


def feature_extraction_layers_1sec() -> List[Layer]:
    return _feature_extraction_layers([8, 16, 32, 64, 64], [16, 16, 16, 16, 31], [2, 2, 2, 2, 4])


def feature_extraction_layers_2sec() -> List[Layer]:
    return _feature_extraction_layers([8, 16, 32, 64, 64], [16, 16, 16, 16, 31], [2, 2, 2, 2, 2])


def feature_extraction_layers_3sec() -> List[Layer]:
    return _feature_extraction_layers([8, 16, 32, 64, 64], [16, 16, 16, 16, 23], [2, 2, 4, 4, 4])


def feature_extraction_layers_5sec() -> List[Layer]:
    return _feature_extraction_layers([8, 16, 32, 64, 64], [16, 16, 16, 16, 39], [2, 4, 4, 4, 4])


def classification_layers(dropout: bool = True, legacy: bool = False) -> List[Layer]:
    if legacy:
        last_layer: Layer = Dense(2, softmax)
    else:
        last_layer: Layer = Dense(1, sigmoid)
    if dropout:
        return [Dense(200, relu), Dropout(0.5), Dense(200, relu), Dropout(0.5), last_layer]
    else:
        return [Dense(200, relu), Dense(200, relu), last_layer]


def layers_1sec() -> List[Layer]:
    return feature_extraction_layers_1sec() + classification_layers()


def layers_2sec() -> List[Layer]:
    return feature_extraction_layers_2sec() + classification_layers()


def layers_3sec() -> List[Layer]:
    return feature_extraction_layers_3sec() + classification_layers()


def layers_5sec() -> List[Layer]:
    return feature_extraction_layers_5sec() + classification_layers()

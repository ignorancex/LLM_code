# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Dict, List, Optional, Tuple, Union

LAYERS_CONTAINER = "layer_container"
N_LAYERS = "n_layers"

FC_UP = "fc_up"
FC_DOWN = "fc_down"
FC_ACT = "fc_act"
FC_GATE = "fc_gate"
MLP = "mlp"

LAYER_TYPES = [FC_UP, FC_DOWN, FC_ACT, FC_GATE, MLP]

MODEL_MAPS: Dict[str, Dict[str, Union[int, str]]] = {
    "dummy": {
        N_LAYERS: 2,
        LAYERS_CONTAINER: "layers",
        FC_UP: "up",
        FC_ACT: "activation_fn",
        FC_DOWN: "down",
        MLP: "",
    },
    "opt-350M": {
        N_LAYERS: 24,
        LAYERS_CONTAINER: "model.decoder.layers",
        FC_UP: "fc1",
        FC_ACT: "activation_fn",
        FC_DOWN: "fc2",
        MLP: "",
    },
    "llama-v3-8B": {
        N_LAYERS: 32,
        LAYERS_CONTAINER: "model.layers",
        FC_UP: "mlp.up_proj",
        FC_GATE: "mlp.gate_proj",
        FC_ACT: "mlp.act_fn",
        FC_DOWN: "mlp.down_proj",
        MLP: "mlp",
    },
    "phi-3-medium": {
        N_LAYERS: 40,
        LAYERS_CONTAINER: "model.layers",
        FC_UP: "mlp.gate_up_proj",
        FC_GATE: "mlp.gate_up_proj",
        FC_ACT: "mlp.activation_fn",
        FC_DOWN: "mlp.down_proj",
        MLP: "mlp",
    },
    "phi-3-mini": {
        N_LAYERS: 32,
        LAYERS_CONTAINER: "model.layers",
        FC_UP: "mlp.gate_up_proj",
        FC_GATE: "mlp.gate_up_proj",
        FC_ACT: "mlp.activation_fn",
        FC_DOWN: "mlp.down_proj",
        MLP: "mlp",
    },
    "turbosparse-mistral": {
        N_LAYERS: 32,
        LAYERS_CONTAINER: "model.layers",
        FC_UP: "mlp.up_proj",
        FC_GATE: "mlp.gate_proj",
        FC_ACT: "mlp.act_fn",
        FC_DOWN: "mlp.down_proj",
        MLP: "mlp",
    },
    "mistral-v01-7B": {
        N_LAYERS: 32,
        LAYERS_CONTAINER: "model.layers",
        FC_UP: "mlp.up_proj",
        FC_GATE: "mlp.gate_proj",
        FC_ACT: "mlp.act_fn",
        FC_DOWN: "mlp.down_proj",
        MLP: "mlp",
    },
}


def block_number_from_id(model_id: str, submodule_id: str) -> int:
    """
    Convert a string corresponding to a block ID to a block number.
    """
    prefix = MODEL_MAPS[model_id][LAYERS_CONTAINER]
    return int(submodule_id.replace(prefix, "").split(".")[1])


def layer_id_to_block_id(model_id: str, submodule_id: str) -> str:
    """
    Convert a string corresponding to a layer ID to a block number.
    """
    block_number = block_number_from_id(model_id, submodule_id)
    return get_block_id(model_id=model_id, block_name=block_number)


def cast_layer_names_to_int_list(model_id: str, layer_names: Union[int, List[int]]) -> List[int]:
    """
    Convert a list of block numbers, or the keyword 'none' or 'all' into a list of integers
    representing the block numbers.
    """
    if isinstance(layer_names, str):
        if layer_names == "all":
            layer_names = range(MODEL_MAPS[model_id][N_LAYERS])
        elif layer_names == "none":
            layer_names = []
        else:
            raise ValueError(
                "'layers_to_sparsify' must be one of 'all', 'none', int or list of ints"
            )
    elif isinstance(layer_names, int):
        layer_names = [layer_names]
    return layer_names


def get_block_id(model_id: str, block_name: int) -> str:
    """
    Convert a block number to a block ID.
    """

    block_id = ".".join(
        [
            MODEL_MAPS[model_id][LAYERS_CONTAINER],
            str(block_name),
        ]
    )

    return block_id


def block_id_to_mlp_id(model_id: str, block_id: str) -> str:
    """
    Convert a block number to a MLP ID.
    """
    mlp_id = MODEL_MAPS[model_id][MLP]
    return block_id if mlp_id == "" else ".".join([block_id, mlp_id])


def block_id_to_layer_ids(model_id: str, block_id: str) -> Tuple[str, str, Optional[str]]:
    """
    Convert a block number to a layer ID.
    """
    down_name = MODEL_MAPS[model_id][FC_DOWN]
    up_name = MODEL_MAPS[model_id][FC_UP]

    down_id = ".".join([block_id, down_name])
    up_id = ".".join([block_id, up_name])
    gate_id = None

    if has_gate(model_id):
        gate_name = MODEL_MAPS[model_id][FC_GATE]
        gate_id = ".".join([block_id, gate_name])

    return up_id, down_id, gate_id


def has_gate(model_id: str):
    """
    Check if a model has a gate layer in the MLPs
    """
    return FC_GATE in MODEL_MAPS[model_id]


def get_block_ids(model_id: str, block_names: Union[int, List[int], str]) -> List[str]:
    """
    Convert a list of block numbers, or the keyword 'none' or 'all' into a list of block IDs
    """
    if isinstance(block_names, str):
        if block_names == "all":
            n_layers: int = MODEL_MAPS[model_id][N_LAYERS]
            block_names = range(n_layers)
        else:
            raise ValueError(f"'{block_names}' must be one of 'all', int or list of ints")
    block_names = cast_layer_names_to_int_list(model_id, block_names)
    return [get_block_id(model_id=model_id, block_name=block_name) for block_name in block_names]


def get_layer_id(model_id: str, layer_type: str, layer_name: int) -> str:
    """
    Convert a layer number to a layer ID.
    """
    if layer_type not in LAYER_TYPES or layer_type is None:
        raise ValueError(f"Layer {layer_type} not found in {LAYER_TYPES}")
    if model_id not in MODEL_MAPS:
        raise ValueError(f"Model {model_id} not found in {MODEL_MAPS.keys()}")
    if layer_type not in MODEL_MAPS[model_id]:
        raise ValueError(f"Layer {layer_type} not specified in {MODEL_MAPS[model_id].keys()}")
    if layer_name > MODEL_MAPS[model_id][N_LAYERS] or layer_name < 0:
        raise ValueError(
            f"Layer {layer_name} out of bounds for net {model_id}, maximum is {MODEL_MAPS[model_id][N_LAYERS]}"
        )

    layer_name = ".".join([get_block_id(model_id, layer_name), MODEL_MAPS[model_id][layer_type]])
    return layer_name


def get_layer_ids(
    model_id: str, layer_type: str, layer_names: Union[str, List[int], int]
) -> List[str]:
    """
    Convert a list of block numbers, or the keyword 'none' or 'all' into a list of layer IDs
    """
    layer_names = cast_layer_names_to_int_list(model_id, layer_names)

    return [
        get_layer_id(model_id=model_id, layer_type=layer_type, layer_name=layer_names)
        for layer_names in layer_names
    ]

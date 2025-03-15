from pathlib import Path
from typing import List, NoReturn, Optional

from tensorflow import is_tensor
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model

import globalconfig as g_conf
from utilities.kerasutils import apply_block
from utilities.typingutils import is_typed_list


def _get_model_files(file_name: str = "", file_extension: str = ".h5", model_path: Optional[Path] = None) \
        -> (Path, Path):
    assert isinstance(file_name, str)
    assert isinstance(file_extension, str)
    assert isinstance(model_path, Path) or model_path is None

    if file_name != "":
        file_name += "_"

    if model_path is None:
        model_path = g_conf.get_models_path()
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)
    assert model_path.is_dir()

    ta: Path = model_path / (file_name + "ta" + file_extension)
    ts: Path = model_path / (file_name + "ts" + file_extension)

    return ta, ts


class SimCLRHelper:
    def __init__(self, ta_input: Input, ts_input: Input, f: List[Layer], g: List[Layer], h: List[Layer],
                 g_keep: int = 0,
                 name: str = "model"):
        """
        A helper class to easily create models for SimCLR type experiments.

        The task-agnostic model has the following form:
            ``model_input -> f -> g``
        while the task-specific model has the following from:
            ``model_input -> f -> g[:g_keep] -> h``

        The names of the layers must be unique across ``f``, ``g``, and ``h``, in order to enable model saving.

        :param ta_input: The input for the task-agnostic model
        :param ts_input: The input for the task-specific model
        :param f: Task-agnostic layers
        :param g: Projection-head layers
        :param h: Task-specific layers
        :param g_keep: Number of projection layers to keep for the task-specific model
        :param name: The model's name (cannot be empty, i.e. "")
        """
        assert is_tensor(ta_input)
        assert is_tensor(ts_input)
        assert is_typed_list(f, Layer)
        assert is_typed_list(g, Layer)
        assert is_typed_list(h, Layer)
        assert isinstance(g_keep, int)
        assert isinstance(name, str)

        self.name: str = name
        self.ta_model: Model = Model(
            inputs=ta_input, outputs=apply_block(f + g, ta_input), name=name + "_ta")
        self.ts_model: Model = Model(
            inputs=ts_input, outputs=apply_block(f + g[:g_keep] + h, ts_input), name=name + "_ts")

    def set_ta_mode(self) -> NoReturn:
        """Set all layers of the task-agnostic model to trainable."""
        for layer in self.ta_model.layers:
            layer.trainable = True

    def set_ts_mode(self) -> NoReturn:
        """Set the task-agnostic and projection-head layers of the task-specific model to not trainable."""
        for layer in self.ta_model.layers:
            layer.trainable = False

    def save_weights(self, model_path: Optional[Path] = None, overwrite: bool = True) -> NoReturn:
        assert isinstance(model_path, Path) or model_path is None
        assert isinstance(overwrite, bool)

        ta, ts = _get_model_files(self.name, model_path=model_path)
        print("Saving models to files:\n  ta: " + str(ta) + "\n  ts: " + str(ts))
        self.ta_model.save_weights(str(ta), overwrite)
        self.ts_model.save_weights(str(ts), overwrite)

    def load_weights(self, model_path: Optional[Path] = None) -> NoReturn:
        assert isinstance(model_path, Path) or model_path is None

        ta, ts = _get_model_files(self.name, model_path=model_path)
        print("Loading models from files:\n  ta: " + str(ta) + "\n  ts: " + str(ts))
        self.ta_model.load_weights(str(ta), True)
        self.ts_model.load_weights(str(ts), True)

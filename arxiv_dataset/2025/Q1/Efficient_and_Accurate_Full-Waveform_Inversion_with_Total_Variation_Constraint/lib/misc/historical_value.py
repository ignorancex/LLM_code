from typing import List, Literal, Union

import numpy as np
import numpy.typing as npt

from lib.misc.colored_text import ColoredText

PreferValueType = Union[Literal["greater"], Literal["less"], None]


# naming is subtle
class ValueHistoryList[T]:
    values: List[T]
    prefer_value: PreferValueType
    label: str

    def __init__(self, label: str, prefer_value: PreferValueType, initial_values=None):
        self.label = label
        self.prefer_value = prefer_value
        self.values = [] if initial_values is None else initial_values

    def append(self, value: T) -> None:
        self.values.append(value)

    def is_prev_improved(self) -> bool:
        if self.prefer_value is None:
            return False

        if len(self.values) < 2:
            return True
        elif self.prefer_value == "greater":
            return self.values[-1] > self.values[-2]
        else:
            return self.values[-1] < self.values[-2]

    def is_prev_improved_arrow_string(self) -> str:
        is_prev_improved = self.is_prev_improved()
        if self.prefer_value is None:
            return ""
        elif self.prefer_value == "greater":
            return ColoredText.green_up() if is_prev_improved else ColoredText.red_down()
        else:
            return ColoredText.green_down() if is_prev_improved else ColoredText.red_up()

    def prev_value_message(self, precision: int) -> str:
        if self.prefer_value is None:
            return f"{self.label}: {self.values[-1]:.{precision}f}"
        else:
            return f"{self.label}: {self.values[-1]:.{precision}f} {self.is_prev_improved_arrow_string()}"

    def values_as_np_array(self) -> npt.NDArray[T]:
        return np.array(self.values)

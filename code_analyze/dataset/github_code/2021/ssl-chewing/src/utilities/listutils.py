from typing import List, Union

import numpy as np

from utilities.numpyutils import is_numpy_1d_vector
from utilities.typingutils import is_typed_list


def select_by_index(lst: List, idxs: Union[List[int], np.ndarray]):
    assert isinstance(lst, List)
    assert is_typed_list(idxs, int) or is_numpy_1d_vector(idxs)

    return [lst[i] for i in idxs]

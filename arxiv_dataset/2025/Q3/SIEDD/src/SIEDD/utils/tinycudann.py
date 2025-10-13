from contextlib import ContextDecorator
from typing import TYPE_CHECKING

try:
    import tinycudann as tcnn  # noqa: F401 # type: ignore
except Exception:
    tcnn = None

if TYPE_CHECKING:
    import tinycudann as tcnn  # noqa: F401 # type: ignore


class requires_tcnn(ContextDecorator):
    def __enter__(self):
        if tcnn is None:
            raise ImportError("tiny-cuda-nn is required")
        return tcnn

    def __exit__(self, *exc):
        return False

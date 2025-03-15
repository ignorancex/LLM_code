from typing import Callable
from pathlib import Path
import onnxruntime as ort
from dunedn.training.metrics import MetricsList


class OnnxNetwork(ort.InferenceSession):
    """Subclass"""

    def __init__(self, ckpt: Path, metrics: MetricsList, providers: list[str] = None):
        """
        Parameters
        ----------
        ckpt: Path
            `.onnx` file path.
        metrics: MetricsList
            List of callable metrics.
        providers: list[str]
            List of providers.
        """
        super().__init__(ckpt, providers=providers)
        self.metrics = metrics

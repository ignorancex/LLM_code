from pathlib import Path
from dunedn.networks.utils import BatchProfiler
import torch
from ..gcnn.gcnn_dataloading import GcnnDataset
from .onnx_abstract_net import OnnxNetwork
from .utils import gcnn_onnx_inference_pass
from dunedn.training.metrics import MetricsList


class OnnxGcnnNetwork(OnnxNetwork):
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
        super().__init__(ckpt, metrics, providers=providers)

    def predict(
        self, generator: GcnnDataset, profiler: BatchProfiler = None
    ) -> torch.Tensor:
        """ONNX GCNN network inference.

        Parameters
        ----------
        generator: GcnnDataset
            The inference generator.
        profiler: BatchProfiler
            The profiler object to record batch inference time.

        Returns
        -------
        torch.Tensor
            Output tensor of shape=(N,C,H,W).
        """
        generator.to_crops()
        test_loader = torch.utils.data.DataLoader(
            dataset=generator,
            batch_size=generator.batch_size,
        )
        output = gcnn_onnx_inference_pass(test_loader, self, profiler=profiler)
        y_pred = generator.converter.tiles2planes(output)
        generator.to_planes()
        return y_pred

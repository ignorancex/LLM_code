from dunedn.networks.utils import BatchProfiler
from tqdm.auto import tqdm
import numpy as np
import torch
import onnxruntime as ort


def gcnn_onnx_inference_pass(
    test_loader: torch.utils.data.DataLoader,
    ort_session: ort.InferenceSession,
    verbose: int = 1,
    profiler: BatchProfiler = None,
) -> torch.Tensor:
    """
    Parameters
    ----------
    generator: torch.utils.data.DataLoader
        The inference dataset generator.
    ort_session: ort.InferenceSession
        The onnxruntime inference session.
    verbose: int
        Switch to log information. Defaults to 1. Available options:

        - 0: no logs.
        - 1: display progress bar.

    profiler: BatchProfiler
            The profiler object to record batch inference time.

    Returns
    -------
    torch.Tensor
        Output tensor of shape=(N,C,H,W).
    """
    outs = []
    wrap = tqdm(test_loader, desc="onnx.predict") if verbose else test_loader
    if profiler is not None:
        wrap = profiler.set_iterable(wrap)
    for noisy, _ in wrap:
        out = ort_session.run(
            None,
            {"input": noisy.numpy().astype(np.float32)},
        )[0]
        outs.append(torch.Tensor(out))
    output = torch.cat(outs)
    return output

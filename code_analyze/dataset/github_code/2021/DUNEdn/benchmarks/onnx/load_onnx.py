"""
    This module generates a dummy batch to be passed to PyTorch and `onnx`
    exported networks for inference time comparison.

    Usage: (assuming being in DUNEdn root folder)
    
    ```
    python benchmarks/onnx/load_onnx.py <modeltype> --onnx <path>
    ```

    Note: the current implementation works only for modeltype cnn | gcnn

    Check that the dummy dataset batch size is consistent with the required input
    shape of the `onnx` model.
"""

from pathlib import Path
import argparse
from time import time as tm
import numpy as np
import torch
import onnx
import onnxruntime as ort
from dunedn.networks.gcnn.gcnn_net import GCNN_Net, USCG_Net
from save_onnx import get_dummy_input_and_gcnn, get_dummy_input_and_uscg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "modeltype", help="model type, available options: uscg | cnn | gcnn"
    )
    parser.add_argument(
        "--onnx", type=Path, help="the saved onnx checkpoint", default="network.onnx"
    )
    parser.add_argument(
        "--batch_size", type=int, help="onnx model batch size", default=32
    )
    parser.add_argument("--check", action="store_true", help="check the onnx model")
    parser.add_argument("--dev", help="device name", default="cpu")
    args = parser.parse_args()

    # dummy dataset
    if args.modeltype == "uscg":
        dummy_input, torch_model = get_dummy_input_and_uscg()
    elif args.modeltype in ["gcnn", "cnn"]:
        dummy_input, torch_model = get_dummy_input_and_gcnn(
            args.modeltype, args.batch_size
        )
    else:
        raise NotImplementedError("Model not implemented")

    print("Network input shape: ", dummy_input.shape)

    if args.check:
        onnx_model = onnx.load(args.onnx.as_posix())
        onnx.checker.check_model(onnx_model)
        print(onnx.helper.printable_graph(onnx_model.graph))

    ort_session = ort.InferenceSession(args.onnx.as_posix())
    start = tm()
    outputs = ort_session.run(
        None,
        {"input": dummy_input.numpy().astype(np.float32)},
    )
    onnx_time = tm() - start
    print(f"ONNX Inference with ONNX done in {onnx_time} s.")

    torch_model.eval()
    dummy_input = torch.Tensor(dummy_input)
    start = tm()
    outputs = torch_model(dummy_input)
    pyt_time = tm() - start
    print(f"PyTorch inference done in {pyt_time} s.")
    print(
        f"ONNX speedup, absolute: {pyt_time/onnx_time:.3}x relative: {100*(pyt_time - onnx_time)/pyt_time:.2}%"
    )

# TODO: add --dev flag to run computation on different devices (cpu / gpu)

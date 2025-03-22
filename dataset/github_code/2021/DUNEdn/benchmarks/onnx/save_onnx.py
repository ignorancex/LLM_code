"""
    This module exports the query denoising network to `onnx` format.

    Usage: (assuming being in DUNEdn root folder)

    ```
    python benchmarks/onnx/save_onnx.py <modeltype> --onnx <path> --dev <device>
    ```

    Note: the current implementation works only for modeltype cnn | gcnn

    It is user responsibility to provide the correct device host through the
    `--dev` flag.

    The onnx export has a fixed input shape. Change the value of the 
    batch_size with `--batch_size` flag according to your needs, default is 32.
"""
from pathlib import Path
import argparse
import torch
from dunedn.networks.gcnn.gcnn_net import USCG_Net, GCNN_Net


def get_dummy_input_and_uscg():
    """
    Returns
    -------
        - torch.Tensor, the dummy input dataset of shape=(1, 1, h, w)
        - USCG_Net, the denoising network
    """
    # USCG Net
    h = 100
    w = 500
    pixels = h * w
    dummy_input = torch.arange(pixels).reshape([1, 1, h, w]) / pixels
    return dummy_input, USCG_Net(h=h, w=w)


def get_dummy_input_and_gcnn(modeltype, batch_size):
    """
    Parameters
    ----------
        - modeltype: str, available options cnn | gcnn
        - batch_size: int, the dummy dataset batch size

    Returns
    -------
        - torch.Tensor, the dummy input dataset of shape=(batch_size, 1, crop_edge, crop_edge)
        - GCNN_Net, the denoising network
    """
    # GCNN Net
    crop_edge = 32
    # edge_size = (edge_h, edge_w)
    pixels = batch_size * crop_edge**2
    dummy_input = (
        torch.arange(pixels).reshape([batch_size, 1, crop_edge, crop_edge]) / pixels
    )
    k = 8 if modeltype == "gcnn" else None
    return dummy_input, GCNN_Net(modeltype, "dn", crop_edge, 1, 32, k=k)


def get_dummy_input_and_model(modeltype, batch_size):
    """
    Parameters
    ----------
        - modeltype: str, available options uscg | cnn | gcnn
        - batch_size: int, the dummy dataset batch size (only for cnn | gcnn)

    Returns
    -------
        - torch.Tensor, dummy data for forward pass
        - torch.nn.Module, the denoising network

    Raises
    ------
        - NotImplementedError if modeltype is not in ['uscg', 'cnn', 'gcnn']
    """
    if modeltype == "uscg":
        return get_dummy_input_and_uscg()
    elif modeltype in ["gcnn", "cnn"]:
        return get_dummy_input_and_gcnn(modeltype, batch_size)
    else:
        raise NotImplementedError("Model not implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "modeltype", help="model type, available options: uscg | cnn | gcnn"
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        help="the output onnx model filename. Default: 'network.onnx'",
        default="network.onnx",
    )
    parser.add_argument(
        "--batch_size", type=int, help="onnx model batch size. Default: 32", default=32
    )
    parser.add_argument("--dev", help="device name. Default: cpu", default="cpu")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print verbose exported model representation",
    )
    args = parser.parse_args()

    if args.modeltype == "uscg" and args.batch_size != 1:
        print(
            f"WARNING: uscg network accepts batch size equal to 1 only: found {args.batch_size}, skipping this"
        )
    dummy_input, model = get_dummy_input_and_model(args.modeltype, args.batch_size)

    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(
        model,
        dummy_input,
        args.onnx,
        verbose=args.verbose,
        input_names=input_names,
        output_names=output_names,
    )

# TODO: add --dev flag to run computation on different devices (cpu / gpu)

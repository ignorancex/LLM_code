"""
    The script version of the ``onnx_example.ipynb`` jupyter notebook.

    Usage

    .. code-block:: bash

        python onnx_example.py -m <modeltype> -v v08 -d cuda:0
    
    Use the optional ``--export`` flag to export the PyTorch model to Onnx
    format.
"""
from pathlib import Path
import argparse
import numpy as np
from assets.functions import (
    prepare_folders_and_paths,
    check_in_output_folder,
    inference,
    plot_example,
)
from dunedn.inference.hitreco import DnModel
from dunedn.inference.analysis import analysis_main
from dunedn.utils.utils import load_runcard

# TODO: the check-in is not complete if runcard do not get copied into the tmp folder


def main(modeltype, version, pytorch_dev, should_export_onnx):
    # base folders
    base_folder = Path("../../output/tmp")
    ckpt_folder = Path(f"../dunedn_checkpoints/{modeltype}_{version}")

    folders, paths = prepare_folders_and_paths(
        modeltype, version, base_folder, ckpt_folder
    )

    check_in_output_folder(folders)

    plot_example(paths["input"], paths["target"], outdir=folders["id_plot"])

    evt = np.load(paths["input"])[:, 2:]
    print(f"Loaded event at {paths['input']}")

    ############################################################################
    # PyTorch
    setup = load_runcard(base_folder / "cards/runcard.yaml")  # settings
    model = DnModel(setup, modeltype, ckpt_folder)
    print(f"Loaded model from {ckpt_folder} folder")

    pytorch_time = inference(model, evt, paths["pytorch"], pytorch_dev)
    print(f"PyTorch inference done in {pytorch_time}s")

    analysis_main(paths["pytorch"], paths["target"])
    plot_example(paths["pytorch"], paths["target"], outdir=folders["pytorch_plot"])

    ############################################################################
    # Onnx
    if should_export_onnx:
        model.onnx_export(folders["onnx_save"])
    model_onnx = DnModel(setup, modeltype, folders["onnx_save"], should_use_onnx=True)
    onnx_time = inference(model_onnx, evt, paths["onnx"])
    print(f"ONNX inference done in {onnx_time}s")

    analysis_main(paths["onnx"], paths["target"])
    plot_example(paths["onnx"], paths["target"], outdir=folders["onnx_plot"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", help="modeltype", dest="modeltype")
    parser.add_argument("--version", "-v", help="model version", default="v08")
    parser.add_argument(
        "--dev", "-d", help="device hosting the computation", default="cpu"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        dest="should_export_onnx",
        help="wether to export to onnx or not",
    )
    args = parser.parse_args()
    main(args.modeltype, args.version, args.dev, args.should_export_onnx)

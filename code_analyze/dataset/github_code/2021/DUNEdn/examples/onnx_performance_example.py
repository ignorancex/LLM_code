"""
    This is the script version of the ``.ipynb`` jupyter notebook.

    Usage

    .. code-block:: bash

        python onnx_performance_example.py -m <modeltype> -v v08 -d cuda:0
    
    Use the optional ``--export`` flag to export the PyTorch model to Onnx
    format.
"""
from pathlib import Path
import argparse
from assets.functions import (
    prepare_folders_and_paths,
    check_in_output_folder,
    compare_performance_onnx,
    plot_comparison_catplot,
)
from dunedn.inference.hitreco import DnModel
from dunedn.utils.utils import load_runcard


def main(modeltype, version, pytorch_dev, should_export_onnx):
    # base folders
    base_folder = Path("../../output/tmp")
    ckpt_folder = Path(f"../dunedn_checkpoints/{modeltype}_{version}")

    folders, paths = prepare_folders_and_paths(
        modeltype, version, base_folder, ckpt_folder
    )

    check_in_output_folder(folders)

    ############################################################################
    # PyTorch
    setup = load_runcard(base_folder / "cards/runcard.yaml")  # settings
    model = DnModel(setup, modeltype, ckpt_folder)
    print(f"Loaded model from {ckpt_folder} folder")

    ############################################################################
    # Onnx
    if should_export_onnx:
        model.onnx_export(folders["onnx_save"])
    model_onnx = DnModel(setup, modeltype, folders["onnx_save"], should_use_onnx=True)

    ############################################################################
    # Speed comparison
    batch_size_list = [32, 64, 128, 256, 512, 1024]
    nb_batches = 2
    performance_df = compare_performance_onnx(
        model, model_onnx, pytorch_dev, batch_size_list, nb_batches
    )
    performance_df.to_csv(paths["performance_csv"])
    plot_comparison_catplot(performance_df, folders["plot"])


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

"""
    This module contains the wrapper function for the ``dunedn preprocess``
    command.

    Example
    -------

    Preprocess help output:

    .. code-block:: text

        $ dunedn preprocess --help
        usage: dunedn preprocess [-h] [--output OUTPUT] [--force] [--save_sample] runcard

        Preprocess dataset of protoDUNE events: dumps planes and training crops.

        positional arguments:
          runcard               the input folder

        optional arguments:
          -h, --help            show this help message and exit
          --output OUTPUT, -o OUTPUT
                                the output folder
          --force               overwrite existing files if present
          --save_sample         extract a smaller dataset
"""
from pathlib import Path
from argparse import ArgumentParser, Namespace
from dunedn.preprocessing.putils import (
    get_planes_and_dump,
    save_normalization_info,
    crop_planes_and_dump,
)
from dunedn.utils.utils import load_runcard, initialize_output_folder, save_runcard


def add_arguments_preprocessing(parser: ArgumentParser):
    """Adds preprocessing subparser arguments.

    Parameters
    ----------
    parser: ArgumentParser
        Preprocessing subparser object.
    """
    parser.add_argument(
        "runcard",
        type=Path,
        help="the input folder",
        default=None,
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="the output folder", default=Path("./data")
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing files if present"
    )
    parser.add_argument(
        "--save_sample", action="store_true", help="extract a smaller dataset"
    )
    parser.set_defaults(func=preprocess)


def preprocess(args: Namespace):
    """Wrapper preprocessing function.

    Parameters
    ----------
    args: Namespace
        Command line parsed arguments. It should contain configcard file name,
        dataset directory path, plus save_sample boolean options.
    """
    setup = load_runcard(args.runcard)
    setup.update({"output": args.output})
    initialize_output_folder(args.output, args.force)
    save_runcard(args.output / "cards/runcard.yaml", setup)
    # save a default runcard in folder to allow default resoration
    save_runcard(args.output / "cards/runcard_default.yaml", setup)

    preprocess_main(
        setup["dataset"],
        args.save_sample,
    )


def preprocess_main(dsetup: dict, save_sample: bool):
    """Preprocessing main function.

    Loads an input event from file, makes inference and saves the ouptut.

    Parameters
    ----------
    dsetup: dict
        The dataset setup.
    save_sample: bool
        Wether to extract a smaller dataset.

        - dir_name: Path, directory path to dataset
        - nb_crops: int, number of crops from each plane
        - crop_edge: int, crop edge size
        - pct: float, signal / background crop balance
    """
    for folder in ["train", "val", "test"]:
        dname = dsetup["data_folder"] / folder
        (dname / "planes").mkdir(parents=True, exist_ok=True)
        if folder == "train":
            (dname / "crops").mkdir(exist_ok=True)
        get_planes_and_dump(dname, save_sample)
    for channel in ["induction", "collection"]:
        save_normalization_info(dsetup["data_folder"], channel)
    crop_planes_and_dump(
        dsetup["data_folder"] / "train",
        dsetup["nb_crops"],
        dsetup["crop_size"],
        dsetup["pct"],
    )

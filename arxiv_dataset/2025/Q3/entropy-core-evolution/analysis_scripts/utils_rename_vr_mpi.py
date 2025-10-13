#!/usr/bin/env python3
"""
utils_rename_vr_mpi.py

Recursively scan a directory for Velociraptor MPI output files ending
with a trailing '.0' and rename them by stripping the '.0' suffix.
"""
import argparse
import logging
from pathlib import Path


def setup_logging() -> logging.Logger:
    """Configure and return a module-level logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(__name__)


def remove_suffix_zero(directory: Path, logger: logging.Logger) -> None:
    """
    Traverse the given directory and rename files ending with '.0'
    by removing the suffix.

    Parameters
    ----------
    directory : Path
        Root directory to scan for subfolders containing MPI output.
    logger : logging.Logger
        Logger for informational messages.
    """
    if not directory.is_dir():
        logger.error("Provided path '%s' is not a directory.", directory)
        return

    for subdir in directory.iterdir():
        if not subdir.is_dir():
            continue
        # Process each file in the subdirectory
        for file_path in subdir.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix == ".0":
                new_name = file_path.stem
                new_path = file_path.with_name(new_name)
                try:
                    file_path.rename(new_path)
                    logger.info("Renamed '%s' to '%s'", file_path.name, new_name)
                except Exception as e:
                    logger.warning("Failed to rename '%s': %s", file_path, e)


def parse_args():
    """Parse command-line arguments for input directory."""
    parser = argparse.ArgumentParser(
        description="Strip trailing '.0' from MPI-generated VR output filenames"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../stf"),
        help="Root directory containing VR MPI output subfolders",
    )
    return parser.parse_args()


def main():
    logger = setup_logging()
    args = parse_args()
    logger.info("Starting rename in directory '%s'", args.input_dir)
    remove_suffix_zero(args.input_dir, logger)
    logger.info("Done processing.")


if __name__ == "__main__":
    main()

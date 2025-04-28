import argparse
from typing import Optional

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from plato.utils import get_abspath


def run_notebook(
    notebook_name: str,
    notebook_path: Optional[str] = None,
) -> None:
    # run notebook programatically

    if notebook_path is None:
        notebook_path = get_abspath() + "notebooks/"

    filepath = notebook_path + notebook_name + ".ipynb"

    # load the notebook
    with open(filepath, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # set up the notebook execution processor
    ep = ExecutePreprocessor()

    try:
        # execute the notebook
        ep.preprocess(nb)

        # save the executed notebook in the same location
        with open(filepath, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"Successfully executed the notebook: {filepath}")

    except Exception as e:
        print(f"Failed to execute the notebook: {e}")


if __name__ == "__main__":
    # set up the argument parser
    parser = argparse.ArgumentParser(
        description="Execute a Jupyter notebook programmatically."
    )
    parser.add_argument(
        "notebook_name",
        type=str,
        help="Name of the Jupyter notebook file to execute (without .ipynb extension).",
    )
    parser.add_argument(
        "--notebook_path",
        type=str,
        default=None,
        help=(
            "Optional path to the directory containing the notebook, defaults to plato "
            "abspath/notebooks."
        ),
    )

    # parse arguments
    args = parser.parse_args()

    # run the notebook
    run_notebook(args.notebook_name, args.notebook_path)

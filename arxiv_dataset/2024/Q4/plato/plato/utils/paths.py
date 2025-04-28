import os


def get_abspath() -> str:
    """
    Get the absolute path of the plato (main) directory.

    Returns
    -------
    str
        Absolute path of the plato (main) directory.
    """

    return (
        os.path.abspath(__file__)[: os.path.abspath(__file__).find("plato")] + "plato/"
    )

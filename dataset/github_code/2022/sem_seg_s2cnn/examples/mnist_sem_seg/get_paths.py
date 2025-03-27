from pathlib import Path


def get_s2cnn_path():
    return str(Path(__file__).parents[2].absolute())


def get_container_path():
    return str(Path(__file__).parent.joinpath("container").absolute())


def get_datasets_path():
    return str(Path(__file__).parent.joinpath("datasets").absolute())


def get_base_path():
    return str(Path(__file__).parent.absolute())


def get_bind_paths():
    paths = [str(Path(__file__).parents[2].absolute())]
    return paths

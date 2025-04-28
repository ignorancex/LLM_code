from omegaconf import OmegaConf


def resolve_path(path: str) -> str:
    path_dict = OmegaConf.create({"__root__": path})
    path_fix = OmegaConf.to_container(path_dict, resolve=True)
    return path_fix["__root__"]

from typing import List, Dict, Any
from pydantic import BaseModel, Field
import yaml


class ExpertConfig(BaseModel):
    source_model: str


class Configuration(BaseModel):
    base_model: str = Field(..., description="The base model to use")
    experts: List[ExpertConfig] = Field(..., description="List of expert models")
    model_kwargs: Dict[str, Any] = Field(
        ..., description="Keyword arguments for the model"
    )
    weights: List[float] = Field(..., description="Weights for the experts")


def read_config(file_path: str) -> Configuration:
    """
    Read a YAML file and return a Configuration instance.

    :param file_path: Path to the YAML configuration file
    :return: Configuration instance
    """
    with open(file_path, "r") as file:
        yaml_data = yaml.safe_load(file)

    # Convert model_kwargs from list of dicts to a single dict
    if isinstance(yaml_data.get("model_kwargs"), list):
        model_kwargs = {}
        for item in yaml_data["model_kwargs"]:
            model_kwargs.update(item)
        yaml_data["model_kwargs"] = model_kwargs

    config = Configuration(**yaml_data)

    # Check base model is in the list of experts or not
    if config.base_model not in [expert.source_model for expert in config.experts]:
        raise ValueError("Base model must be one of the experts")

    return config

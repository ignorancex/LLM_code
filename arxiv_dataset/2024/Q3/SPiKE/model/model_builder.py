"""
Module for creating models based on configuration.
"""

from model import spike

def create_model(config, num_coord_joints):
    """Create a model based on the config and return it."""
    if "ITOP" in config["dataset"]:
        try:
            model_name = config.get("model")
            if not hasattr(spike, model_name):
                raise ValueError(f"Model {model_name} not found.")

            model_class = getattr(spike, config["model"])
            model_params = {
                "radius": config.get("radius"),
                "nsamples": config.get("nsamples"),
                "spatial_stride": config.get("spatial_stride"),
                "dim": config.get("dim"),
                "depth": config.get("depth"),
                "heads": config.get("heads"),
                "dim_head": config.get("dim_head"),
                "mlp_dim": config.get("mlp_dim"),
                "num_coord_joints": num_coord_joints,
                "dropout1": config.get("dropout1"),
                "dropout2": config.get("dropout2"),
            }

            return model_class(**model_params)
        except ValueError as e:
            print(f"Error: {e}")
    return None

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mod.config import read_config


def test_config():
    config_path = (
        "configs/1.5B/mod_config.yml"  # Replace with your actual YAML file path
    )
    config = read_config(config_path)

    print("Configuration loaded successfully:")
    print(config.model_dump_json(indent=2))

    # You can now access the configuration data like this:
    print(f"\nBase Model: {config.base_model}")
    print(f"Number of Experts: {len(config.experts)}")
    print(f"Model Keywords: {config.model_kwargs}")
    print(f"Weights: {config.weights}")

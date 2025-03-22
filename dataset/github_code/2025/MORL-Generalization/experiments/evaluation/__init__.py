import json
import yaml
import os
from typing import Optional

# def get_eval_params(name: Optional[str] = None):
#     config_file = os.path.join(os.path.dirname(__file__), "eval_params.yaml")
#     with open(config_file) as file:
#         params = yaml.load(file, Loader=yaml.FullLoader)

#     if name and name in params:
#         return params[name]
#     elif name and name not in params:
#         return None
    
#     return params

def get_minmax_values(name: Optional[str] = None):
    config_file = os.path.join(os.path.dirname(__file__), "minmax_values", f"{name}.json")
    config_file = os.path.abspath(config_file)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Normalization boundaries not found for {name}, \
                                set normalization:False in --generalization-hyperparams or provide the file")

    with open(config_file, "r") as f:
        params = json.load(f)

    
    return params

if __name__ == "__main__":
    # print(get_eval_params("MOHumanoidDR-v5"))
    print(get_minmax_values("MOHumanoidDR-v5"))
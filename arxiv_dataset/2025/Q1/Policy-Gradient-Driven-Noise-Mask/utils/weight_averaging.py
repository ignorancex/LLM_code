import torch
from pathlib import Path
from typing import List, Dict
from functools import reduce
from operator import add

def load_model(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location=torch.device('cpu'))['model']

def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        key: reduce(add, (sd[key] for sd in state_dicts)) / len(state_dicts)
        for key in state_dicts[0].keys()
    }

def main():
    model_paths = [f"../model_{i}.pth" for i in range(87, 90)]
    state_dicts = map(load_model, model_paths)
    
    averaged_state_dict = average_state_dicts(list(state_dicts))
    
    output_path = Path("../_weights/## OUTPUT FILE NAME ##")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(averaged_state_dict, output_path)

if __name__ == "__main__":
    main()

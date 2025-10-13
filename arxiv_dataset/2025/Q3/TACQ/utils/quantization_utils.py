import torch 
import os
import yaml
def cross_tensor_sum(masks):
    numerls = 0
    for key in masks:
        numerls += torch.sum(masks[key]).item()
    return numerls

def importances_to_mask_top_p_sparse(
    args,
    importances: dict[str, torch.Tensor],
    fraction: float,
    n_params: int
):
    """
    For a given dictionary of 2D weight tensors, this function selects the top 
    fraction of weights based on importances matrix.
    """

    first_key = next(iter(importances))
    device = importances[first_key].device
    dtype = importances[first_key].dtype

    shapes = {}
    offsets = {}
    total_elems = 0
    for k, v in importances.items():
        shapes[k] = v.shape
        offsets[k] = total_elems
        total_elems += v.numel()

    full_importances = torch.zeros(total_elems, device=device, dtype=dtype)

    for k, v in importances.items():
        start = offsets[k]
        end = start + v.numel()
        full_importances[start:end] = v.reshape(-1)

    print("Importances flattened")

    k_select = int(fraction * n_params)
    k_select = min(k_select, total_elems)  # just in case fraction*n_params > total_elems

    if k_select == 0:
        # Early exit if no elements are to be selected
        return {k: torch.zeros_like(v, dtype=torch.bool) for k, v in importances.items()}

    _, top_indices = torch.topk(full_importances, k_select)

    print("Top indices selected")

    flat_mask = torch.zeros_like(full_importances, dtype=torch.bool)
    flat_mask[top_indices] = True

    # Re-map the mask to original tensor shapes
    masks = {}
    for k, v in importances.items():
        start = offsets[k]
        end = start + v.numel()
        masks[k] = flat_mask[start:end].view(shapes[k])

    print("VERIFY: K_selected |Covered parameters:", k_select, torch.sum(flat_mask))
    return masks


def count_params(model):
    if isinstance(model, dict):
        numerls = 0
        for name, param in model.items():
            n_params = param.numel()
            numerls += n_params
    else:
        numerls = 0
        for name, param in model.named_parameters():
            n_params = param.numel()
            numerls += n_params
    return numerls


def save_important_mask(args, mask: dict[str, torch.Tensor], save_path=None):
    """
    Save a dictionary of boolean PyTorch tensors to a file with special file naming, ensuring efficient storage.
    Each value in `mask` is converted to a boolean tensor (if not already) and moved to CPU 
    before saving to minimize storage size and dependencies.
    """
    if not save_path:
        output_path = os.path.join(args.results_dir, args.run_name, "_important_mask.pt")
    else:
        output_path = save_path
    
    # Convert all values to boolean CPU tensors for efficient storage
    save_mask = {k: v.to(torch.bool).cpu() for k, v in mask.items()}
    
    # Save the mask dictionary in binary format
    torch.save(save_mask, save_path)
    return output_path



def make_quantization_config(args, all_keys, selected_keys, configuration="q3_8", save_path=None):
    """Map a quantization level to each weight name, useful if sdynamically assigning bit-width to modules."""
    if configuration == "q4_8":
        quantization_levels = {}
        for key in all_keys:
            if key in selected_keys:
                quantization_levels[key] = 8
            else:
                quantization_levels[key] = 4
    elif configuration == "q3_8":
        quantization_levels = {}
        for key in all_keys:
            if key in selected_keys:
                quantization_levels[key] = 8
            else:
                quantization_levels[key] = 3
    elif configuration == "q2_8":
        quantization_levels = {}
        for key in all_keys:
            if key in selected_keys:
                quantization_levels[key] = 8
            else:
                quantization_levels[key] = 2
    elif configuration == "q8":
        quantization_levels = {}
        for key in all_keys:
            quantization_levels[key] = 8
    elif configuration == "q4":
        quantization_levels = {}
        for key in all_keys:
            quantization_levels[key] = 4
    elif configuration == "q3":
        quantization_levels = {}
        for key in all_keys:
            quantization_levels[key] = 3
    elif configuration == "q2":
        quantization_levels = {}
        for key in all_keys:
            quantization_levels[key] = 2
    yaml_str = yaml.dump(quantization_levels, 
                        default_flow_style=False,  # Use block style formatting
                        sort_keys=False,           # Preserve dictionary order
                        allow_unicode=True)        # Support unicode characters
    
    # Save to file if path provided
    if not save_path:
        save_path = os.path.join(args.results_dir, args.run_name, f"{args.serial_number}+{configuration}.yaml")
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(yaml_str)
    return yaml_str
import sys
import torch
from safetensors.torch import save_file, load_file
from huggingface_hub import save_torch_state_dict

if len(sys.argv) != 5:
    print(f"Combines two models so that certain layers come from each")
    print("Usage: python stack_layers.py ckpt1_path num_layers_from_ckpt1 ckpt2_path out_file")
    exit()

ckpt1_path = sys.argv[1]
num_layers_from_ckpt1 = int(sys.argv[2])
ckpt2_path = sys.argv[3]
out_model_path = sys.argv[4]

def load_checkpoint(path):
    print("Loading file...", path)
    if path.lower().endswith('.safetensors'):
        return load_file(path, device='cpu')
    else:
        return torch.load(path, map_location='cpu', weights_only=True)

ckpt1 = load_checkpoint(ckpt1_path)
ckpt2 = load_checkpoint(ckpt2_path)

def del_layer_range(ckpt, min_layer_id, end_layer_id):
    for k in list(ckpt.keys()):
        if k.startswith("model.layers."):
            layer_id = int(k.split('.')[2])
            if layer_id >= min_layer_id and layer_id < end_layer_id:
                print("Removing", k)
                del ckpt[k]

print("Removing extraneous layers from ckpt1")
del_layer_range(ckpt1, num_layers_from_ckpt1, 99999)
print("Removing extraneous layers from ckpt2")
del_layer_range(ckpt2, 0, num_layers_from_ckpt1)

print("Creating output state dict")
# add remaining entries in ckpt2 into ckpt1
ckpt1.update(ckpt2)

# save ckpt1
if out_model_path.endswith('.safetensors'):
    print("Saving file...")
    save_file(ckpt1, out_model_path, metadata=dict(format='pt'))
else:
    print("Saving model chunks...")
    save_torch_state_dict(ckpt1, out_model_path)
print("Done!")

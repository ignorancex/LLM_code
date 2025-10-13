# Export script
import torch
from collections import OrderedDict

def export_encoder(path_to_checkpoint, output_path):
    # Load full checkpoint
    checkpoint = torch.load(path_to_checkpoint)
    
    # Extract just the encoder state dict
    full_state_dict = checkpoint['encoder']  # or however your checkpoint is structured
    
    # Create a clean state dict with just encoder weights
    encoder_state_dict = OrderedDict()
    for k, v in full_state_dict.items():
        # Assuming your encoder weights start with 'encoder.'
        if k.startswith('_orig_mod.'):
            # Remove the 'encoder.' prefix for cleaner loading
            k = k.replace('_orig_mod.', '')
        encoder_state_dict[k] = v
    
    # Create metadata/config
    config = {
        "spatial_dims": 3,
        "in_channels": 1,
        "features": (64, 128, 256, 512, 768),
        "act": "GELU",
        "norm": "instance",
        "bias": True,
        "dropout": 0.2
    }
    
    # Save both weights and config
    torch.save({
        "config": config,
        "state_dict": encoder_state_dict
    }, output_path)

# Export each variant
export_encoder('3d-cnn-simclr-mprage/checkpoint.pt', 'base_encoder.pt')
export_encoder('3d-cnn-simclr-bloch/checkpoint.pt', 'seqaug_encoder.pt')
export_encoder('3d-cnn-simclr-bloch-paired/checkpoint.pt', 'seqinv_encoder.pt')

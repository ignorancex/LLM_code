import torch
import argparse
from model import CNNEncoder, ViTEncoder

def extract_encoder_weights(unet_checkpoint_path, encoder_type, output_path, wandb_id, epoch, metric):
    # Load the full UNet checkpoint
    checkpoint = torch.load(unet_checkpoint_path, map_location='cpu')
    
    # Create an instance of the appropriate encoder
    if encoder_type == 'cnn':
        encoder = CNNEncoder(
            spatial_dims=3, 
            in_channels=1, 
            features=(64, 128, 256, 512, 768), 
            act="GELU", 
            norm="instance", 
            bias=True, 
            dropout=0.2
        )
    elif encoder_type == 'vit':
        encoder = ViTEncoder(
            spatial_dims=3,
            in_channels=1,
            img_size=96,  # Adjust this if needed
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            dropout_rate=0.2,
            qkv_bias=True,
            save_attn=False
        )
    else:
        raise ValueError("Unsupported encoder type. Choose 'cnn' or 'vit'.")

    # Extract encoder weights from the UNet state dict
    encoder_state_dict = {}
    for key, value in checkpoint['model'].items():
        if key.startswith('conv_0'):
            encoder_state_dict[key] = value
        elif key.startswith('down_'):
            encoder_state_dict[key] = value

    # Load the extracted weights into the encoder
    encoder.load_state_dict(encoder_state_dict)

    class WandBID:
        def __init__(self, wandb_id):
            self.wandb_id = wandb_id

        def state_dict(self):
            return self.wandb_id

    class Epoch:
        def __init__(self, epoch):
            self.epoch = epoch

        def state_dict(self):
            return self.epoch

    class Metric:
        def __init__(self, metric):
            self.metric = metric

        def state_dict(self):
            return self.metric

    # Save the encoder weights to a new checkpoint file
    torch.save({
        'encoder': encoder.state_dict(),
        'encoder_type': encoder_type,
        'wandb': WandBID(wandb_id).state_dict(),
        'epoch': Epoch(epoch).state_dict(),
        'metric': Metric(metric).state_dict()
    }, output_path)

    print(f"Encoder weights extracted and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract encoder weights from a UNet checkpoint")
    parser.add_argument("unet_checkpoint", type=str, help="Path to the UNet checkpoint file")
    parser.add_argument("encoder_type", type=str, choices=['cnn', 'vit'], help="Type of encoder (cnn or vit)")
    parser.add_argument("output_path", type=str, help="Path to save the extracted encoder weights")
    parser.add_argument("--wandb_id", type=str, help="WANDB ID")
    parser.add_argument("--epoch", type=int, help="Epoch")
    parser.add_argument("--metric", type=float, help="Metric")
    
    args = parser.parse_args()

    extract_encoder_weights(args.unet_checkpoint, args.encoder_type, args.output_path, args.wandb_id, args.epoch, args.metric)

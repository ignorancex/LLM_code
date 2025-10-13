import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn

class FullyFusedDeformNetwork(nn.Module):
    """
    Deformation network using tiny-cuda-nn's FullyFusedMLP for high-performance training and inference.
    This is a drop-in replacement for the standard DeformNetwork in utils/time_utils.py
    """
    def __init__(self, D=16, W=128, input_ch=3, output_ch=10, multires=12):
        super(FullyFusedDeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6
        self.skips = [D // 2]

        # Calculate input dimensions for positional encoding (matching standard MLP)
        # 3D coordinates: 3 * (1 + 2 * multires) = 3 * (1 + 2 * 10) = 63
        # Time: 1 * (1 + 2 * t_multires) = 1 * (1 + 2 * 4) = 9
        # Total: 63 + 9 = 72
        xyz_input_ch = 3 * (1 + 2 * multires)  # 63
        time_input_ch = 1 * (1 + 2 * self.t_multires)  # 9
        total_input_ch = xyz_input_ch + time_input_ch  # 72

        # tiny-cuda-nn config for FullyFusedMLP with Frequency encoding
        config = {
            "encoding": {
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,  # 3D coordinates
                        "otype": "Frequency",
                        "n_frequencies": multires  # 10 frequencies
                    },
                    {
                        "n_dims_to_encode": 1,  # Time
                        "otype": "Frequency",
                        "n_frequencies": self.t_multires  # 4 frequencies
                    }
                ]
            },
            "network": {
                "otype": "FullyFusedMLP",
                "n_neurons": W,
                "n_hidden_layers": D - 1,  # Now 11 hidden layers instead of 7
                "activation": "ReLU",
                "output_activation": "None"
            }
        }

        self.tcnn_model = tcnn.NetworkWithInputEncoding(
            n_input_dims=4,  # 3D coords + time (raw input)
            n_output_dims=10,  # 3 pos + 4 rot + 3 scale
            encoding_config=config["encoding"],
            network_config=config["network"]
        )
        self.tcnn_model.jit_fusion = False

    def forward(self, x, t):
        input_tensor = torch.cat([x, t], dim=-1)
        output = self.tcnn_model(input_tensor)
        
        # Add residual connection to improve gradient flow (similar to skip connections)
        # Scale the residual to prevent it from dominating the output
        residual_scale = 0.1
        # Create new tensor instead of modifying in-place
        output_with_residual = output.clone()
        # Only add residual to the position output (first 3 dimensions)
        output_with_residual[:, :3] = output[:, :3] + residual_scale * x
        
        d_xyz = output_with_residual[:, :3]
        d_rotation = output_with_residual[:, 3:7]
        d_scaling = output_with_residual[:, 7:10]
        return d_xyz, d_rotation, d_scaling

def create_deform_network(use_fullyfused=False, **kwargs):
    if use_fullyfused:
        try:
            return FullyFusedDeformNetwork(**kwargs)
        except ImportError:
            print("Warning: tinycudann not available, falling back to standard MLP")
            from utils.time_utils import DeformNetwork
            return DeformNetwork(**kwargs)
    else:
        from utils.time_utils import DeformNetwork
        return DeformNetwork(**kwargs) 
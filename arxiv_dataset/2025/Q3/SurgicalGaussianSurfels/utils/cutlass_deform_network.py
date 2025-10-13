import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
from utils.fullyfused_deform_network import create_deform_network as create_fullyfused_deform_network


class CutlassDeformNetwork(nn.Module):
    """
    Deformation network using tiny-cuda-nn's CutlassMLP for high-performance training and inference.
    This is a drop-in replacement for the standard DeformNetwork in utils/time_utils.py
    """
    
    def __init__(self, D=8, W=256, input_ch=3, output_ch=10, multires=10):
        super(CutlassDeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 4
        self.skips = [D // 2]

        # Calculate input dimensions for positional encoding (matching standard MLP)
        # 3D coordinates: 3 * (1 + 2 * multires) = 3 * (1 + 2 * 10) = 63
        # Time: 1 * (1 + 2 * t_multires) = 1 * (1 + 2 * 4) = 9
        # Total: 63 + 9 = 72
        xyz_input_ch = 3 * (1 + 2 * multires)  # 63
        time_input_ch = 1 * (1 + 2 * self.t_multires)  # 9
        total_input_ch = xyz_input_ch + time_input_ch  # 72

        # tiny-cuda-nn configuration for CutlassMLP with Frequency encoding
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
                "otype": "CutlassMLP",
                "n_neurons": W,
                "n_hidden_layers": D - 1,
                "activation": "ReLU",
                "output_activation": "None"
            }
        }

        # Create tiny-cuda-nn model with proper input dimensions
        self.tcnn_model = tcnn.NetworkWithInputEncoding(
            n_input_dims=4,  # 3D coords + time (raw input)
            n_output_dims=10,  # 3 pos + 4 rot + 3 scale
            encoding_config=config["encoding"],
            network_config=config["network"]
        )

        # Disable JIT fusion to avoid compatibility issues
        # self.tcnn_model.jit_fusion = tcnn.supports_jit_fusion()
        self.tcnn_model.jit_fusion = False

    def forward(self, x, t):
        """
        Forward pass through the CutlassMLP deformation network.
        
        Args:
            x: 3D coordinates tensor [N, 3]
            t: Time tensor [N, 1]
            
        Returns:
            d_xyz: Position offsets [N, 3]
            d_rotation: Rotation quaternions [N, 4] 
            d_scaling: Scaling factors [N, 3]
        """
        # Ensure inputs are float32 and on GPU
        x = x.float().cuda()
        t = t.float().cuda()
        
        # Concatenate 3D coordinates and time
        input_tensor = torch.cat([x, t], dim=-1)
        
        # Ensure input tensor is contiguous
        input_tensor = input_tensor.contiguous()
        
        # Forward through tiny-cuda-nn model
        output = self.tcnn_model(input_tensor)
        
        # Add residual connection to improve gradient flow (similar to skip connections)
        # Scale the residual to prevent it from dominating the output
        residual_scale = 0.1
        # Create new tensor instead of modifying in-place
        output_with_residual = output.clone()
        # Only add residual to the position output (first 3 dimensions)
        output_with_residual[:, :3] = output[:, :3] + residual_scale * x
        
        # Split output into components
        d_xyz = output_with_residual[:, :3]
        d_rotation = output_with_residual[:, 3:7]
        d_scaling = output_with_residual[:, 7:10]
        
        return d_xyz, d_rotation, d_scaling


def create_deform_network(use_cutlass=False, use_fullyfused=False, **kwargs):
    """
    Factory function to create deformation network.
    
    Args:
        use_cutlass: If True, use CutlassMLP; otherwise use standard PyTorch MLP
        use_fullyfused: If True, use FullyFusedMLP
        **kwargs: Arguments passed to the network constructor
        
    Returns:
        Deformation network instance
    """
    if use_fullyfused:
        return create_fullyfused_deform_network(use_fullyfused=True, **kwargs)
    if use_cutlass:
        try:
            return CutlassDeformNetwork(**kwargs)
        except ImportError:
            print("Warning: tinycudann not available, falling back to standard MLP")
            from utils.time_utils import DeformNetwork
            return DeformNetwork(**kwargs)
        except Exception as e:
            print(f"Warning: CutlassMLP failed with error: {e}")
            print("Falling back to standard MLP")
            from utils.time_utils import DeformNetwork
            return DeformNetwork(**kwargs)
    else:
        from utils.time_utils import DeformNetwork
        return DeformNetwork(**kwargs) 
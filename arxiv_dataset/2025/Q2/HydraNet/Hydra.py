import torch
import torch.nn as nn
import torch.nn.functional as F


def silu(x):
    return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, z):
        x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Hydra(nn.Module):
    def __init__(self, d_model: int,
                 n_layer: int = 1,
                 d_state: int = 128,
                 expand: int = 2,
                 headdim: int = 32,
                 ):
        super().__init__()
        self.n_layer = n_layer
        self.d_state = d_state
        self.headdim = headdim
        self.d_inner = expand * d_model
        assert self.d_inner % self.headdim == 0, "self.d_inner must be divisible by headdim"
        self.nheads = self.d_inner // self.headdim

        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        self.dt_bias = nn.Parameter(torch.empty(self.nheads, ))
        self.A_log = nn.Parameter(torch.randn(self.nheads) * 0.01)  # Initialize as small random numbers
        self.D = nn.Parameter(torch.ones(self.nheads) * 0.01)  # Initialize close to zero
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.reduce_layer = nn.Linear(2 * self.headdim, self.headdim)  # Define a dimensionality reduction linear layer

        # Define cross attention and fusion layers as member variables
        self.cross_attention = nn.MultiheadAttention(embed_dim=2 * self.headdim, num_heads=self.nheads,
                                                     batch_first=True)
        self.fusion_layer = nn.Linear(2 * self.d_inner, self.d_inner)

    def forward(self, u: torch.Tensor):
        """
        u: [B, L, d_model]
        """
        A = -torch.exp(self.A_log)  # [nheads]
        zxbcdt = self.in_proj(u)  # [B, L, d_in_proj]
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,  # z part [B, L, d_inner]
                self.d_inner + 2 * self.d_state,  # xBC part [B, L, d_inner + 2*d_state]
                self.nheads,  # dt part [B, L, nheads]
            ],
            dim=-1,
        )

        dt = F.softplus(dt + self.dt_bias)  # [B, L, nheads]

        x, B_feat, C_feat = torch.split(
            xBC, [self.d_inner, self.d_state, self.d_state], dim=-1
        )  # [B, L, d_inner], [B, L, d_state], [B, L, d_state]

        B_feat = B_feat.unsqueeze(2)  # [B, L, 1, d_state]
        C_feat = C_feat.unsqueeze(2)  # [B, L, 1, d_state]

        # Reshape x for multi-head attention
        B_, L, H, P = x.shape[0], x.shape[1], self.nheads, self.headdim
        x = x.view(B_, L, H, P)  # [B, L, H, P]

        y = self.ssd(x * dt.unsqueeze(-1),
                     A * dt,  # [B, L, H]
                     B_feat,  # [B, L, d_state, 1]
                     C_feat)  # [B, L, d_state, 1]

        y = y + x * self.D.unsqueeze(-1)  # Add residual connection

        _b, _l, _h, _p = y.shape  # Get the shape of y
        y = y.reshape(_b, _l, _h * _p)  # Restore shape to (batch, seq_len, d_inner)

        y = self.norm(y, z)  # [B, num_windows, d_inner]
        y = self.out_proj(y)  # [B, num_windows, d_model]

        # Extract last window to update momentum
        last_ball = y[:, -1, :]  # [B, d_model]
        return y, last_ball

    def segsum(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(-1)
        device = x.device
        # x = x.unsqueeze(-2).repeat(1, 1, T, 1)  # [B, H, T, H]
        x = x[..., None].repeat(1, 1, 1, 1, T)  # Repeat along the time dimension
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-1)

        # Enhance numerical stability
        max_val = x_segsum.max(dim=-1, keepdim=True).values.detach()
        x_segsum = (x_segsum - max_val).clamp(min=-50, max=50)

        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def ssd(self, x, A, B, C):
        """
        x: [B, L, H, P]
        A: [B, L, H]
        B: [B, L, d_state, 1]
        C: [B, L, d_state, 1]
        """
        B_, L, H, P = x.shape
        window_size = 2
        stride = 1
        num_windows = L - window_size + 1

        if num_windows < 1:
            raise ValueError("Input sequence length must be at least window_size=2.")

        # Sliding window partition
        x_windows = x.unfold(1, window_size, stride)  # [B, num_windows, H, P, window_size]
        x_windows = x_windows.permute(0, 1, 4, 2, 3)  # Rearrange dimensions to [B, num_windows, window_size, H, P]

        A_windows = A.unfold(1, window_size, stride)  # [B, num_windows, H, window_size]
        A_windows = A_windows.permute(0, 1, 3, 2)  # Rearrange to [B, num_windows, window_size, H]
        A_windows = A_windows.permute(0, 3, 1, 2)  # Change order of dimensions

        B_windows = B.unfold(1, window_size, stride)  # [B, num_windows, H, P, window_size]
        B_windows = B_windows.permute(0, 1, 4, 2, 3)  # Rearrange dimensions

        C_windows = C.unfold(1, window_size, stride)  # [B, num_windows, H, P, window_size]
        C_windows = C_windows.permute(0, 1, 4, 2, 3)  # Rearrange dimensions

        # 1. Compute the diagonal output within each block (Y_diag)
        A_cumsum = torch.cumsum(A_windows, dim=-1)  # [B, num_windows, window_size, H]

        A_segsum = self.segsum(A_windows)
        max_A = A_segsum.max(dim=-1, keepdim=True).values.detach()  # Separate gradient
        L_matrix = torch.exp(A_segsum - max_A)  # Stabilized exponential

        # Compute Y_diag
        Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C_windows, B_windows, L_matrix, x_windows)
        # [B, num_windows, window_size, H, P]

        # 2. Compute states within each block
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # Decay states
        # [B, num_windows, window_size, H]
        states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B_windows, decay_states, x_windows)
        # [B, num_windows, d_state, H, P]

        # 3. Block-to-block learning - recursive block-level SSM for boundary update
        initial_states = torch.zeros_like(states[:, :1, :, :, :])  # [B,1,d_state,H,P]
        states = torch.cat([initial_states, states], dim=1)  # [B, num_windows+1, d_state, H, P]

        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))[0]  # Inter-block decay
        new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)  # Update states across blocks
        states = new_states[:, :-1]  # Remove extra part after update

        # 4. Convert block states to output (Y_off)
        max_A_cumsum = A_cumsum.max(dim=-1, keepdim=True).values.detach()
        state_decay_out = torch.exp(A_cumsum - max_A_cumsum)  # Stabilized exponent
        # state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C_windows, states, state_decay_out)

        # Cross Attention Fusion between Y_diag and Y_off
        Y_diag = self.restore_sequence(Y_diag)
        Y_off = self.restore_sequence(Y_off)
        Y = self.cross_attention_fusion(Y_diag, Y_off)  # [B, num_windows, d_inner]

        return Y  # [B, num_windows, H, P]

    def restore_sequence(self, x):
        """
        Restore the original sequence from windowed representation.
        Input:
            x: [batch_size, num_windows, window_size, num_heads, head_dim]
        Output:
            restored: [batch_size, sequence_length, num_heads, head_dim]
        """
        B, num_windows, window_size, num_heads, head_dim = x.shape
        sequence_length = num_windows + window_size - 1  # Calculate restored sequence length

        # Create an empty tensor to store the restored sequence
        restored = torch.zeros(B, sequence_length, num_heads, head_dim, device=x.device)

        # Count overlaps at each position for averaging
        overlap_count = torch.zeros(sequence_length, device=x.device)

        # Iterate over windows, add window values to corresponding sequence positions
        for i in range(num_windows):
            restored[:, i:i + window_size, :, :] += x[:, i, :, :, :]  # Accumulate window values
            overlap_count[i:i + window_size] += 1  # Increment overlap count

        # Average overlapping parts
        restored /= overlap_count.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # Broadcast to [B, L, H, D]

        return restored

    def cross_attention_fusion(self, Y_diag, Y_off):
        """
        Y_diag, Y_off: [B, num_windows, H, P]
        """
        B, num_windows, H, P = Y_diag.shape

        # Keep original shape and operate directly
        # Reshape for multi-head attention (retain [H, P] structure)
        Y_diag_reshaped = Y_diag.permute(0, 2, 1, 3).reshape(B * H, num_windows, P)  # [B * H, num_windows, P]
        Y_off_reshaped = Y_off.permute(0, 2, 1, 3).reshape(B * H, num_windows, P)  # [B * H, num_windows, P]

        # Concatenate Y_diag and Y_off along feature dimension
        combined = torch.cat([Y_diag_reshaped, Y_off_reshaped], dim=-1)  # [B * H, num_windows, 2 * P]

        # Apply cross attention
        fused, _ = self.cross_attention(combined, combined, combined)  # [B * H, num_windows, 2 * P]

        fused = self.reduce_layer(fused)  # [B * H, num_windows, P]

        # Project back to original dimensions
        fused = fused.reshape(B, H, num_windows, P).permute(0, 2, 1, 3)  # [B, num_windows, H, P]

        return fused  # [B, num_windows, H, P]

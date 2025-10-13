import torch
import torch.nn.functional as F

from core.config import DEVICE



def decompose_from_nmf_basis(input_tensor, W):
    # Given tensor and NMF basis
    #input_tensor = torch.rand(1, c, h, w)  # (1, c, h, w)
    #W = torch.rand(c, k)  # NMF basis of shape (c, k)
    input_tensor = input_tensor.to(DEVICE)
    W = W.to(DEVICE).permute(1,0)
    b, c, h, w = input_tensor.shape
    k = W.shape[1]

    #print("W.shape: {}".format(W.shape))
    #print("input tensor shape: {}".format(input_tensor.shape))

    # Flatten the spatial dimensions of input_tensor
    input_flattened = input_tensor.view(1, c, -1)  # shape (1, c, h * w)

    # Initialize coefficients H with random non-negative values
    H = torch.rand(k, h * w, requires_grad=True, device=DEVICE).float()  # shape (k, h * w)

    # Set up optimizer to find H that minimizes reconstruction error
    optimizer = torch.optim.Adam([H], lr=0.01)

    # Optimize the coefficients H
    for step in range(500):  # Increase steps as needed
        optimizer.zero_grad()
        # Compute the reconstruction error
        reconstruction = W @ H  # shape (c, h * w)
        #print("reconstruction shape: {}".format(reconstruction.shape))
        #print("input flattened shpae: {}".format(input_flattened.shape))
        loss = F.mse_loss(reconstruction, input_flattened.squeeze(0))  # Loss between reconstruction and input
        #print(loss)
        loss.backward()
        optimizer.step()

        # Optional: Apply non-negativity constraint on H (clamp to non-negative values)
        with torch.no_grad():
            H.clamp_(min=0)

    # Reshape H to match (k, h, w) if needed
    H = H.view(k, h, w)  # Final decomposed representation in NMF basis

    # Compute final reconstruction in original shape
    final_reconstruction = (W @ H.view(k, h * w)).view(1, c, h, w)

    return H, final_reconstruction
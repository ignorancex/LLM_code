import torch

def coral(source, target):
    d = source.size(1)  # dim vector (feature dimension)

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    # Frobenius norm of the difference between covariance matrices
    loss = torch.norm(source_c - target_c, p='fro') ** 2

    # Normalize the loss by the squared number of features
    loss = loss / (4 * d * d)
    return loss


def compute_covariance(input_data):
    """
    Compute covariance matrix of the input data.
    """
    n = input_data.size(0)  # batch size
    d = input_data.size(1)  # feature dimension
    device = input_data.device

    # # Compute the mean for each feature
    # mean = input_data.mean(dim=0, keepdim=True)

    # # Center the data by subtracting the mean
    # centered_data = input_data - mean

    # # Compute covariance matrix
    # covariance = torch.matmul(centered_data, centered_data.transpose(1, 2)) / (n - 1)

    dt_d = input_data.transpose(2, 3) @ input_data

    one_t_d = (torch.ones(size=input_data.shape, device=device).transpose(2, 3) @ input_data)

    C = (dt_d - (one_t_d.transpose(2, 3) @ one_t_d)/n)/(n - 1)

    return C

import torch.nn as nn
import torch


def frobenius_dist(t1):
    """
    Compute the Frobenius distance for a tensor.
    """
    dot_prod = t1 * t1
    return torch.sqrt(torch.sum(dot_prod, dim=1))


def soft_binning_einsum(x, centers, sigma=0.5):
    """
    Perform soft binning using einsum.
    """
    x = x.unsqueeze(-1)

    centers = centers.to(x.device)

    exponent = -((x - centers) ** 2) / (2 * sigma ** 2)

    return torch.exp(exponent)


def shift_image(x, d, theta):
    """
    Shift the image tensor based on the given angle (theta).
    """
    if theta == 0:
        result = torch.roll(x, shifts=-d, dims=3)
        result[:, :, :, -d:, :] = 0
    elif theta == 45:
        result = torch.roll(x, shifts=(d, -d), dims=(2, 3))
        result[:, :, 0:d, :, :] = 0
        result[:, :, :, -d:, :] = 0
    elif theta == 90:
        result = torch.roll(x, shifts=d, dims=2)
        result[:, :, 0:d, :, :] = 0
    elif theta == 135:
        result = torch.roll(x, shifts=(d, d), dims=(2, 3))
        result[:, :, 0:d, :, :] = 0
        result[:, :, :, 0:d, :] = 0
    else:
        raise ValueError("Invalid theta value. Must be one of [0, 45, 90, 135].")
    return result


def soft_binned_glcm_einsum_approx(x, d, theta, num_levels, min_r=-1, max_r=1):
    """
    Approximate the GLCM using soft binning and einsum.
    """
    centers = torch.linspace(min_r, max_r, num_levels)
    I_bins = soft_binning_einsum(x, centers)  # Shape: [batch, channels, height, width, num_levels]
    I_s_bins = shift_image(I_bins, d, theta)
    occurrences = torch.einsum('bchwj,bchwk->bcjk', I_bins, I_s_bins)
    glcm = occurrences + occurrences.permute(0, 1, 3, 2)
    glcm_sum = glcm.sum(dim=(2, 3), keepdim=True)
    glcm_sum = torch.where(glcm_sum == 0, torch.ones_like(glcm_sum), glcm_sum)
    glcm /= glcm_sum
    return glcm


def compute_haralick_features(glcm):
    """
    Compute Haralick features from the GLCM.
    """
    num_gray_levels = glcm.shape[2]
    I = torch.arange(0, num_gray_levels).unsqueeze(1).to(glcm.device)  # Column vector
    J = torch.arange(0, num_gray_levels).unsqueeze(0).to(glcm.device)  # Row vector
    weights = (I - J) ** 2
    weights = weights.reshape((1, 1, num_gray_levels, num_gray_levels)).to(glcm.device)
    contrast = torch.sum(glcm * weights, dim=(2, 3))

    return contrast


def _extract_grid(image):
    """
    Extract a grid of Haralick features from the image.
    """
    haralick_grid = []
    for i in [1, 3, 5, 7]:
        haralick_grid.append(compute_haralick_features(
            soft_binned_glcm_einsum_approx(image, d=i, theta=0, num_levels=256, min_r=-1, max_r=1)))
    for i in [1, 2, 4, 6]:
        haralick_grid.append(compute_haralick_features(
            soft_binned_glcm_einsum_approx(image, d=i, theta=45, num_levels=256, min_r=-1, max_r=1)))
    for i in [1, 3, 5, 7]:
        haralick_grid.append(compute_haralick_features(
            soft_binned_glcm_einsum_approx(image, d=i, theta=90, num_levels=256, min_r=-1, max_r=-1)))
    for i in [1, 2, 4, 6]:
        haralick_grid.append(compute_haralick_features(
            soft_binned_glcm_einsum_approx(image, d=i, theta=135, num_levels=256, min_r=-1, max_r=-1)))
    return torch.cat(haralick_grid, dim=0).view(image.size(0), 1, 4, 4)


def _texture_loss(fake_im, real_im, opt, grid_extractor, model=None):
    """
    Compute the texture loss between fake and real images.
    """
    textures_real = grid_extractor(real_im)
    textures_fake = grid_extractor(fake_im)
    delta_grids = (torch.abs(textures_fake - textures_real)).view(opt.batch_size, -1)

    if opt.texture_criterion == 'max':
        criterion_texture, _ = torch.max(delta_grids, dim=1)
    elif opt.texture_criterion == 'average':
        criterion_texture = torch.mean(delta_grids, dim=1)
    elif opt.texture_criterion == 'Frobenius':
        criterion_texture = frobenius_dist(delta_grids)
    elif opt.texture_criterion == 'attention':
        delta_grids = delta_grids.view(opt.batch_size, 1, 4, 4)
        normalized_criterion = (delta_grids - delta_grids.min()) / (delta_grids.max() - delta_grids.min())
        out_attention, map, weight = model(normalized_criterion)
        loss_cycle_texture = torch.abs(torch.mean(torch.sum(out_attention, dim=(2, 3))))
        return loss_cycle_texture, map, weight
    else:
        raise ValueError("Invalid texture criterion. Must be one of ['max', 'average', 'Frobenius', 'attention'].")

    loss_cycle_texture = torch.mean(criterion_texture)
    return loss_cycle_texture


class _GridExtractor(nn.Module):
    """
    Grid extractor module for extracting Haralick features.
    """

    def forward(self, x):
        return _extract_grid(x)


class Self_Attn(nn.Module):
    """
    Self-attention layer for dynamic aggregation of texture features.
    """

    def __init__(self, in_dim, activation='relu'):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass for the self-attention layer.
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out, attention, self.gamma


if __name__ == '__main__':
    """Standard GLCM vs Soft GLCM"""
    import skimage as ski
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from skimage.feature import graycomatrix, graycoprops

    # Load the image and convert to tensor
    img = ski.data.coins()
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)

    # Compute standard GLCM
    glcm_s = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    standard_contrast = graycoprops(glcm_s, "contrast")[0][0]

    # Compute soft GLCM
    soft_glcm = soft_binned_glcm_einsum_approx(img_tensor, d=1, theta=0, num_levels=256, min_r=0, max_r=256)
    soft_contrast = compute_haralick_features(soft_glcm).item()

    # Display results
    print(f"Standard GLCM Contrast: {standard_contrast}")
    print(f"Soft GLCM Contrast: {soft_contrast}")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Standard GLCM")
    plt.imshow(glcm_s[:, :, 0, 0], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("Soft GLCM")
    plt.imshow(soft_glcm[0, 0, :, :].cpu().numpy(), cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("Difference")
    plt.imshow(abs(torch.tensor(glcm_s[:, :, 0, 0]) - soft_glcm[0, 0, :, :]).cpu().numpy(), cmap="gray")
    plt.show()

    """Texture Loss computation: dummy example"""

    fake_im = torch.rand((1, 1, 256, 256))  # Batch size 1, 1 channel, 256x256 image
    real_im = torch.rand((1, 1, 256, 256))  # Batch size 1, 1 channel, 256x256 image

    class Opt:
        def __init__(self, texture_criterion):
            self.batch_size = 1
            self.texture_criterion = texture_criterion

    grid_extractor = _GridExtractor()
    attention_model = Self_Attn(in_dim=1)  # Assuming the input dimension is 1 for the attention model

    def compute_texture_loss(criterion):
        opt = Opt(texture_criterion=criterion)
        if criterion == 'attention':
            loss, attention_map, weights = _texture_loss(fake_im, real_im, opt, grid_extractor, attention_model)
            print(f"Computed Texture Loss ({criterion.capitalize()} Criterion): {loss.item()}")
        else:
            loss = _texture_loss(fake_im, real_im, opt, grid_extractor)
            print(f"Computed Texture Loss ({criterion.capitalize()} Criterion): {loss.item()}")


    # Example usage
    for criterion in ['max', 'average', 'Frobenius', 'attention']:
        compute_texture_loss(criterion)

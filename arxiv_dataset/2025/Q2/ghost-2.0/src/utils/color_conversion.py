#from https://github.com/kornia/kornia/tree/main/kornia/color

import math

import torch
import torch.nn as nn
import numpy as np

def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Convert an image from RGB to HSV.

    .. image:: _static/img/rgb_to_hsv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        eps: scalar to enforce numarical stability.

    Returns:
        HSV version of the image with shape of :math:`(*, 3, H, W)`.
        The H channel values are in the range 0..2pi. S and V are in the range 0..1.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, argmin_rgb = image.min(-3)
    deltac = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

    h1 = (bc - gc)
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h = (h / 6.0) % 1.0
    h = 2. * math.pi * h  # we return 0/2pi output

    return torch.stack((h, s, v), dim=-3)



def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from HSV to RGB.

    The H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.

    Args:
        image: HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    h: torch.Tensor = image[..., 0, :, :] / (2 * math.pi)
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)

    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)

    return out


def linear_rgb_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a linear RGB image to sRGB. Used in colorspace conversions.

    Args:
        image: linear RGB Image to be converted to sRGB of shape :math:`(*,3,H,W)`.

    Returns:
        sRGB version of the image with shape of shape :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = linear_rgb_to_rgb(input) # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    threshold = 0.0031308
    rgb: torch.Tensor = torch.where(
        image > threshold, 1.055 * torch.pow(image.clamp(min=threshold), 1 / 2.4) - 0.055, 12.92 * image
    )

    return rgb

def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to XYZ.

    .. image:: _static/img/rgb_to_xyz.png

    Args:
        image: RGB Image to be converted to XYZ with shape :math:`(*, 3, H, W)`.

    Returns:
         XYZ version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_xyz(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    x: torch.Tensor = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y: torch.Tensor = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z: torch.Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out: torch.Tensor = torch.stack([x, y, z], -3)

    return out


def xyz_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a XYZ image to RGB.

    Args:
        image: XYZ Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = xyz_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    x: torch.Tensor = image[..., 0, :, :]
    y: torch.Tensor = image[..., 1, :, :]
    z: torch.Tensor = image[..., 2, :, :]

    r: torch.Tensor = 3.2404813432005266 * x + -1.5371515162713185 * y + -0.4985363261688878 * z
    g: torch.Tensor = -0.9692549499965682 * x + 1.8759900014898907 * y + 0.0415559265582928 * z
    b: torch.Tensor = 0.0556466391351772 * x + -0.2040413383665112 * y + 1.0573110696453443 * z

    out: torch.Tensor = torch.stack([r, g, b], dim=-3)

    return out

def rgb_to_linear_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an sRGB image to linear RGB. Used in colorspace conversions.

    .. image:: _static/img/rgb_to_linear_rgb.png

    Args:
        image: sRGB Image to be converted to linear RGB of shape :math:`(*,3,H,W)`.

    Returns:
        linear RGB version of the image with shape of :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_linear_rgb(input) # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    lin_rgb: torch.Tensor = torch.where(image > 0.04045, torch.pow(((image + 0.055) / 1.055), 2.4), image / 12.92)

    return lin_rgb

def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to Lab.

    .. image:: _static/img/rgb_to_lab.png

    The image data is assumed to be in the range of :math:`[0, 1]`. Lab
    color is computed using the D65 illuminant and Observer 2.

    Args:
        image: RGB Image to be converted to Lab with shape :math:`(*, 3, H, W)`.

    Returns:
        Lab version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_lab(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)

    xyz_im: torch.Tensor = rgb_to_xyz(lin_rgb)

    # normalize for D65 white point
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    threshold = 0.008856
    power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = torch.where(xyz_normalized > threshold, power, scale)

    x: torch.Tensor = xyz_int[..., 0, :, :]
    y: torch.Tensor = xyz_int[..., 1, :, :]
    z: torch.Tensor = xyz_int[..., 2, :, :]

    L: torch.Tensor = (116.0 * y) - 16.0
    a: torch.Tensor = 500.0 * (x - y)
    _b: torch.Tensor = 200.0 * (y - z)

    out: torch.Tensor = torch.stack([L, a, _b], dim=-3)

    return out




def lab_to_rgb(image: torch.Tensor, clip: bool = True) -> torch.Tensor:
    r"""Convert a Lab image to RGB.

    Args:
        image: Lab image to be converted to RGB with shape :math:`(*, 3, H, W)`.
        clip: Whether to apply clipping to insure output RGB values in range :math:`[0, 1]`.

    Returns:
        Lab version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = lab_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    L: torch.Tensor = image[..., 0, :, :]
    a: torch.Tensor = image[..., 1, :, :]
    _b: torch.Tensor = image[..., 2, :, :]

    fy = (L + 16.0) / 116.0
    fx = (a / 500.0) + fy
    fz = fy - (_b / 200.0)

    # if color data out of range: Z < 0
    fz = fz.clamp(min=0.0)

    fxyz = torch.stack([fx, fy, fz], dim=-3)

    # Convert from Lab to XYZ
    power = torch.pow(fxyz, 3.0)
    scale = (fxyz - 4.0 / 29.0) / 7.787
    xyz = torch.where(fxyz > 0.2068966, power, scale)

    # For D65 white point
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz.device, dtype=xyz.dtype)[..., :, None, None]
    xyz_im = xyz * xyz_ref_white

    rgbs_im: torch.Tensor = xyz_to_rgb(xyz_im)

    # https://github.com/richzhang/colorization-pytorch/blob/66a1cb2e5258f7c8f374f582acc8b1ef99c13c27/util/util.py#L107
    #     rgbs_im = torch.where(rgbs_im < 0, torch.zeros_like(rgbs_im), rgbs_im)

    # Convert from RGB Linear to sRGB
    rgb_im = linear_rgb_to_rgb(rgbs_im)

    # Clip to 0,1 https://www.w3.org/Graphics/Color/srgb
    if clip:
        rgb_im = torch.clamp(rgb_im, min=0.0, max=1.0)

    return rgb_im

def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    .. image:: _static/img/rgb_to_yuv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b

    out: torch.Tensor = torch.stack([y, u, v], -3)

    return out


def yuv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YUV image to RGB.

    The image data is assumed to be in the range of (0, 1) for luma and (-0.5, 0.5) for chroma.

    Args:
        image: YUV Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = yuv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: torch.Tensor = image[..., 0, :, :]
    u: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    r: torch.Tensor = y + 1.14 * v  # coefficient for g is 0
    g: torch.Tensor = y + -0.396 * u - 0.581 * v
    b: torch.Tensor = y + 2.029 * u  # coefficient for b is 0

    out: torch.Tensor = torch.stack([r, g, b], -3)

    return out


#from https://github.com/pytorch/vision/issues/598
def color_transfer(
    input: torch.Tensor,
    source: torch.Tensor,
    mode: str = "pca",
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Transfer the colors from one image tensor to another, so that the target image's
    histogram matches the source image's histogram. Applications for image histogram
    matching includes neural style transfer and astronomy.

    The source image is not required to have the same height and width as the target
    image. Batch and channel dimensions are required to be the same for both inputs.

    Gatys, et al., "Controlling Perceptual Factors in Neural Style Transfer", arXiv, 2017.
    https://arxiv.org/abs/1611.07865

    Args:

        input (torch.Tensor): The NCHW or CHW image to transfer colors from source
            image to from the source image.
        source (torch.Tensor): The NCHW or CHW image to transfer colors from to the
            input image.
        mode (str): The color transfer mode to use. One of 'pca', 'cholesky', or 'sym'.
            Default: "pca"
        eps (float): The desired epsilon value to use.
            Default: 1e-5

    Returns:
        matched_image (torch.tensor): The NCHW input image with the colors of source
            image. Outputs should ideally be clamped to the desired value range to
            avoid artifacts.
    """

    assert input.dim() == 3 or input.dim() == 4
    assert source.dim() == 3 or source.dim() == 4
    input = input.unsqueeze(0) if input.dim() == 3 else input
    source = source.unsqueeze(0) if source.dim() == 3 else source
    assert input.shape[:2] == source.shape[:2]

    # Handle older versions of PyTorch
    torch_cholesky = (
        torch.linalg.cholesky if torch.__version__ >= "1.9.0" else torch.cholesky
    )

    def torch_symeig_eigh(x: torch.Tensor):
        """
        torch.symeig() was deprecated in favor of torch.linalg.eigh()
        """
        if torch.__version__ >= "1.9.0":
            L, V = torch.linalg.eigh(x, UPLO="U")
        else:
            L, V = torch.symeig(x, eigenvectors=True, upper=True)
        return L, V

    def get_mean_vec_and_cov(
        x_input: torch.Tensor, eps: float
    ):
        """
        Convert input images into a vector, subtract the mean, and calculate the
        covariance matrix of colors.
        """
        x_mean = x_input.mean(3).mean(2)[:, :, None, None]

        # Subtract the color mean and convert to a vector
        B, C = x_input.shape[:2]
        # x_vec = (x_input - x_mean).reshape(B, C, -1)
        x_vec = x_input.reshape(B, C, -1)
        # Calculate covariance matrix
        x_cov = torch.bmm(x_vec, x_vec.permute(0, 2, 1)) / x_vec.shape[2]
        # This line is only important if you get artifacts in the output image
        # x_cov = x_cov + (eps * torch.eye(C, device=x_input.device)[None, :])
        return torch.zeros_like(x_mean), x_vec, x_cov
        # return x_mean, x_vec, x_cov
    
    def pca(x: torch.Tensor) -> torch.Tensor:
        """Perform principal component analysis"""
        eigenvalues, eigenvectors = torch_symeig_eigh(x)
        e = torch.sqrt(torch.diag_embed(eigenvalues.reshape(eigenvalues.size(0), -1)))
        # Remove any NaN values if they occur
        if torch.isnan(e).any():
            e = torch.where(torch.isnan(e), torch.zeros_like(e), e)
        return torch.bmm(torch.bmm(eigenvectors, e), eigenvectors.permute(0, 2, 1))

    # Collect & calculate required values
    _, input_vec, input_cov = get_mean_vec_and_cov(input, eps)
    source_mean, _, source_cov = get_mean_vec_and_cov(source, eps)

    # Calculate new cov matrix for input
    if mode == "pca":
        new_cov = torch.bmm(pca(source_cov), torch.inverse(pca(input_cov)))
    elif mode == "cholesky":
        new_cov = torch.bmm(
            torch_cholesky(source_cov), torch.inverse(torch_cholesky(input_cov))
        )
    elif mode == "sym":
        p = pca(input_cov)
        pca_out = pca(torch.bmm(torch.bmm(p, source_cov), p))
        new_cov = torch.bmm(torch.bmm(torch.inverse(p), pca_out), torch.inverse(p))
    else:
        raise ValueError(
            "mode has to be one of 'pca', 'cholesky', or 'sym'."
            + " Received '{}'.".format(mode)
        )

    # Multiply input vector by new cov matrix
    new_vec = torch.bmm(new_cov, input_vec)

    # Reshape output vector back to input's shape &
    # add the source mean to our output vector
    return new_vec.reshape(input.shape) + source_mean



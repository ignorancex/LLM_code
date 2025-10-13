import torch
import einops
from ..configs import PosEncoderType, CoordXConfig
from fractions import Fraction


def pad_to_patch_size(img_tensor, patch_height, patch_width):
    """
    Pads the image tensor to make it divisible by the patch dimensions.

    Parameters:
    img_tensor (torch.Tensor): Tensor of shape [batch_size, channels, height, width].
    patch_height (int): Height of each patch.
    patch_width (int): Width of each patch.

    Returns:
    torch.Tensor: Padded image tensor.
    """
    batch_size, channels, height, width = img_tensor.shape

    pad_height = (patch_height - height % patch_height) % patch_height
    pad_width = (patch_width - width % patch_width) % patch_width

    # Pad the right and bottom sides of the image
    padded_img = torch.nn.functional.pad(
        img_tensor, (0, pad_width, 0, pad_height), mode="constant", value=0
    )

    return padded_img


def patchify_images(img_tensor, patch_height, patch_width):
    """
    Split a batch of images into patches using einops.

    Parameters:
    img_tensor (torch.Tensor): Tensor of shape [batch_size, channels, height, width].
    patch_height (int): Height of each patch.
    patch_width (int): Width of each patch.

    Returns:
    torch.Tensor: Tensor of patches with shape [batch_size, num_patches, channels, patch_height, patch_width].
    """
    batch_size, channels, height, width = img_tensor.shape

    if height % patch_height != 0 or width % patch_width != 0:
        # Pad the images to make them divisible by the patch size
        img_tensor = pad_to_patch_size(img_tensor, patch_height, patch_width)

    # Use einops.rearrange to split the images into patches
    patches = einops.rearrange(
        img_tensor,
        "b c (h ph) (w pw) -> b (h w) c ph pw",
        ph=patch_height,
        pw=patch_width,
    )

    return patches


def unpatchify_images(
    patches: torch.Tensor,
    original_height: int,
    original_width: int,
    patch_height: int,
    patch_width: int,
    strided: bool = False,
):
    """
    Reconstruct images from their patches and remove any padding.

    Parameters:
    patches (torch.Tensor): Tensor of patches with shape [batch_size, num_patches, channels, patch_height, patch_width].
    original_height (int): Original height of the image before patchifying.
    original_width (int): Original width of the image before patchifying.
    patch_height (int): Height of each patch.
    patch_width (int): Width of each patch.

    Returns:
    torch.Tensor: Reconstructed image tensor of shape [batch_size, channels, original_height, original_width].
    """

    # Calculate the height and width after patchifying (before unpadding)
    padded_height = (
        (original_height + patch_height - 1) // patch_height
    ) * patch_height
    padded_width = ((original_width + patch_width - 1) // patch_width) * patch_width

    if strided:
        order_h, order_w = "(ph h)", "(pw w)"
    else:
        order_h, order_w = "(h ph)", "(w pw)"
    if patches.ndim == 6:
        """ When we have both patch and group sizes"""
        reconstructed = einops.rearrange(
            patches,
            f"b g (h w) c ph pw -> (b g) c {order_h} {order_w}",
            h=padded_height // patch_height,
            w=padded_width // patch_width,
        )

    else:
        if patches.ndim == 4:
            patches = patches.unsqueeze(0)
        # Unpatchify the image
        reconstructed = einops.rearrange(
            patches,
            f"b (h w) c ph pw -> b c {order_h} {order_w}",
            h=padded_height // patch_height,
            w=padded_width // patch_width,
        )

    # Crop the image to remove any padding, restoring original dimensions
    reconstructed = reconstructed[:, :, :original_height, :original_width]

    return reconstructed


def generate_coordinates(
    input_shape: list[int],
    patch_size: int | None,
    normalize_range: tuple[float, float],
    positional_encoding: PosEncoderType,
    patch_scale: Fraction | None = None,
):
    """
    Generate coordinates for a given input shape.

    :param input_shape: Tuple representing the shape of the input.
    :param patch_size: Return centroid coordinates for each patch of size patch_size.
    :param normalize_range: Range to normalize the coordinates to.
    :return: A list or array of coordinates.
    """
    coordinates: torch.Tensor | list[torch.Tensor]
    if len(input_shape) == 2:
        H, W = input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError()

    if isinstance(positional_encoding, CoordXConfig) and patch_size is not None:
        ph = pw = patch_size
        centroids_h = torch.arange(ph // 2, H, ph)[:, None]
        centroids_w = torch.arange(pw // 2, W, pw)[:, None]
        coordinates = [centroids_h, centroids_w]
    elif isinstance(positional_encoding, CoordXConfig):
        coordinates = [torch.arange(H)[:, None], torch.arange(W)[:, None]]
    elif patch_size is None:
        coordinates = einops.rearrange(
            einops.repeat(torch.arange(0, H), "h -> h w", w=W), "h w -> (h w) 1"
        )
        coordinates = torch.cat(
            [
                coordinates,
                einops.rearrange(
                    einops.repeat(torch.arange(0, W), "w -> h w", h=H), "h w -> (h w) 1"
                ),
            ],
            dim=1,
        )
    else:
        ph = pw = patch_size
        centroids_h = torch.arange(0, H, ph)
        centroids_w = torch.arange(0, W, pw)
        # Create grid of centroids
        grid_h, grid_w = torch.meshgrid(centroids_h, centroids_w, indexing="ij")
        coordinates = einops.rearrange(
            torch.stack([grid_h, grid_w], dim=-1), "h w c -> (h w) c"
        )

    if isinstance(positional_encoding, CoordXConfig):
        coordinates_list = [coordinates[0] / (H - 1), coordinates[1] / (W - 1)]
        return [
            c * (normalize_range[1] - normalize_range[0]) + normalize_range[0]
            for c in coordinates_list
        ]
    elif isinstance(coordinates, torch.Tensor):
        coordinates = coordinates / torch.tensor([H - 1, W - 1])
        coordinates = (
            coordinates * (normalize_range[1] - normalize_range[0]) + normalize_range[0]
        )
        if patch_scale is not None:
            # coordinates is now (N, 3) instead of (N, 2)
            patch_scales = torch.full(
                (coordinates.shape[0], 1), float(patch_scale), device=coordinates.device
            )
            coordinates = torch.cat((coordinates, patch_scales), dim=-1)
        return coordinates
    else:
        raise ValueError()

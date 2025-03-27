"""
    This module contains the utility functions for CNN and GCNN networks.
"""
from typing import Tuple
import torch
import torch.nn.functional as F


def pairwise_dist(arr, k, local_mask):
    """
    Computes pairwise euclidean distances between pixels.

    Parameters
    ----------
        - arr: torch.Tensor, of shape=(N,H*W,C)
        - k: int, nearest neighbor number
        - local_mask: torch.Tensor, of shape=(1,H*W,H*W)

    Returns
    -------
        - torch.Tensor, pairwise pixel distances of shape=(N,H*W,H*W)
    """
    dev = arr.get_device()
    dev = "cpu" if dev == -1 else dev
    local_mask = local_mask.to(dev)
    r_arr = torch.sum(arr * arr, dim=2, keepdim=True)  # (B,N,1)
    mul = torch.matmul(arr, arr.permute(0, 2, 1))  # (B,N,N)
    d = -(r_arr - 2 * mul + r_arr.permute(0, 2, 1))  # (B,N,N)
    d = d * local_mask - (1 - local_mask)
    del mul, r_arr
    # this is the euclidean distance wrt the feature vector of the current pixel
    # then the matrix has to be of shape (B,N,N), where N=prod(crop_size)
    return d.topk(k=k, dim=-1)[1]  # (B,N,K)


def batched_index_select(t, dim, inds):
    """
    Selects K nearest neighbors indices for each pixel respecting batch dimension.

    Parameters
    ----------
        - t: torch.Tensor of shape=(N,H*W,C)
        - dim: int, pixels axis. Default is 1
        - inds: torch.Tensor, neighbors indices of shape=(N,H*W,C)

    Returns
    -------
        - torch.Tensor, index tensor of shape=(N,H*W*K,C)
    """
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    return t.gather(dim, dummy)


def local_mask(crop_size):
    """
    Computes mask to remove local pixels from the computation.

    Parameters
    ----------
        - crops_size: tuple, (H,W)

    Returns
    -------
        - torch.Tensor, local mask of shape=(1,H*W,H*W)
    """
    x, y = crop_size
    nb_pixels = x * y

    local_mask = torch.ones([nb_pixels, nb_pixels])
    for ii in range(nb_pixels):
        if ii == 0:
            local_mask[ii, (ii + 1, ii + y, ii + y + 1)] = 0  # top-left
        elif ii == nb_pixels - 1:
            local_mask[ii, (ii - 1, ii - y, ii - y - 1)] = 0  # bottom-right
        elif ii == x - 1:
            local_mask[ii, (ii - 1, ii + y, ii + y - 1)] = 0  # top-right
        elif ii == nb_pixels - x:
            local_mask[ii, (ii + 1, ii - y, ii - y + 1)] = 0  # bottom-left
        elif ii < x - 1 and ii > 0:
            local_mask[
                ii, (ii + 1, ii - 1, ii + y - 1, ii + y, ii + y + 1)
            ] = 0  # first row
        elif ii < nb_pixels - 1 and ii > nb_pixels - x:
            local_mask[
                ii, (ii + 1, ii - 1, ii - y - 1, ii - y, ii - y + 1)
            ] = 0  # last row
        elif ii % y == 0:
            local_mask[
                ii, (ii + 1, ii - y, ii + y, ii - y + 1, ii + y + 1)
            ] = 0  # first col
        elif ii % y == y - 1:
            local_mask[
                ii, (ii - 1, ii - y, ii + y, ii - y - 1, ii + y - 1)
            ] = 0  # last col
        else:
            local_mask[
                ii,
                (
                    ii + 1,
                    ii - 1,
                    ii - y,
                    ii - y + 1,
                    ii - y - 1,
                    ii + y,
                    ii + y + 1,
                    ii + y - 1,
                ),
            ] = 0
    return local_mask.unsqueeze(0)


def calculate_pad(plane_size, crop_size):
    """
    Given plane and crop shape, compute the needed padding to obtain exact
    tiling.

    Parameters
    ----------
        - plane_size: tuple, plane shape (N,C,H,W)
        - crop_size: tuple, crop shape (edge_h, edge_w)

    Returns
    -------
        - list, plane padding: [pre h, post h, pre w, post w]
    """
    return_pad = [0, 0, 0, 0]
    _, _, im_h, im_w = plane_size
    edge_h, edge_w = crop_size

    if (edge_h - (im_h % edge_h)) % 2 == 0:
        return_pad[2] = (edge_h - (im_h % edge_h)) // 2
        return_pad[3] = (edge_h - (im_h % edge_h)) // 2
    else:
        return_pad[2] = (edge_h - (im_h % edge_h)) // 2
        return_pad[3] = (edge_h - (im_h % edge_h)) // 2 + 1

    if (edge_w - (im_w % edge_w)) % 2 == 0:
        return_pad[0] = (edge_w - (im_w % edge_w)) // 2
        return_pad[1] = (edge_w - (im_w % edge_w)) // 2
    else:
        return_pad[0] = (edge_w - (im_w % edge_w)) // 2
        return_pad[1] = (edge_w - (im_w % edge_w)) // 2 + 1
    return return_pad


class Converter:
    """Groups image to tiles converter functions"""

    def __init__(self, crop_size: Tuple[int]):
        """
        Parameters
        ----------
        crop_size: Tuple[int]
            The tile dimensions: (edge_h, edge_w).
        """
        self.crop_size = crop_size

    def planes2tiles(self, planes: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        planes: torch.Tensor
            Planes, of shape=(N,C,W,H).

        Returns
        -------
        torch.Tensor
            Tiles of shape=(N',C,edge_h,edge_w).
            With ``N' = N * ceil(H/edge_h) * ceil(W/edge_w)``
        """
        edge_h, edge_w = self.crop_size
        nb_channels = planes.shape[1]
        self.pad = calculate_pad(planes.shape, self.crop_size)
        planes = F.pad(planes, self.pad, mode="constant", value=planes.mean())

        splits = torch.stack(torch.split(planes, edge_w, -1), 1)
        splits = torch.stack(torch.split(splits, edge_h, -2), 1)

        self.splits_shape = splits.shape

        return splits.view(-1, nb_channels, edge_h, edge_w)

    def tiles2planes(self, splits: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        splits: torch.Tensor
            Tiles, of shape (N',C,edge_h,edge_w).

        Returns
        -------
        torch.Tensor
            Planes, of shape=(N,C,H,W).
        """
        b, a_x, a_y, nb_channels, p_x, p_y = self.splits_shape
        nb_channels = splits.shape[1]
        splits_shape = (b, a_x, a_y, nb_channels, p_x, p_y)

        splits = splits.reshape(splits_shape)
        splits = splits.permute(0, 1, 4, 3, 2, 5)
        img = splits.reshape(-1, a_x * p_x, nb_channels, a_y * p_y)
        img = img.permute(0, 2, 1, 3)

        output_shape = (
            slice(None),
            slice(None),
            slice(self.pad[-2], -self.pad[-1]),
            slice(self.pad[0], -self.pad[1]),
        )

        return img[output_shape]

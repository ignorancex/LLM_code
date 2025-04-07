import torch
import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    binary_erosion,
    generate_binary_structure,
)
from utils.pytuils import AverageMeter


class Metric:
    def __init__(self) -> None:
        self.dice = AverageMeter()
        self.jaccard = AverageMeter()
        self.assd = AverageMeter()
        self.hd95 = AverageMeter()

    def add(self, result: torch.Tensor, reference: torch.Tensor) -> None:
        self.dice.add(*dice(result, reference))
        self.jaccard.add(*jaccard(result, reference))
        self.assd.add(*assd(result, reference))
        self.hd95.add(*hd95(result, reference))

    def __str__(self) -> str:
        return f"dice:\t\t{self.dice.get()*100:.2f}\njaccard:\t{self.jaccard.get()*100:.2f}\nassd:\t\t{self.assd.get():.2f}\nhd95:\t\t{self.hd95.get():.2f}"

    def __repr__(self) -> str:
        return self.__str__()


def dice(
    result: torch.Tensor, reference: torch.Tensor, epsilon=1e-6
) -> tuple[float, int]:
    assert result.shape == reference.shape, f"{result.shape}, {reference.shape}"
    assert result.dim() == 4, result.shape

    intersection = torch.sum(result * reference, dim=(2, 3))
    sum_mask = torch.sum(result, dim=(2, 3))
    sum_target = torch.sum(reference, dim=(2, 3))
    dice_score = ((2 * intersection) + epsilon) / (sum_mask + sum_target + epsilon)

    return torch.nansum(dice_score).item(), torch.sum(~torch.isnan(dice_score)).item()


def jaccard(result: torch.Tensor, reference: torch.Tensor) -> tuple[float, int]:
    assert result.shape == reference.shape, f"{result.shape}, {reference.shape}"

    epsilon = 1e-6

    intersection = torch.sum(result * reference, dim=(2, 3))
    sum_mask = torch.sum(result, dim=(2, 3))
    sum_target = torch.sum(reference, dim=(2, 3))
    jaccard_score = ((intersection) + epsilon) / (
        sum_mask + sum_target - intersection + epsilon
    )

    return (
        torch.nansum(jaccard_score).item(),
        torch.sum(~torch.isnan(jaccard_score)).item(),
    )


def surface_distances(
    result: np.ndarray, reference: np.ndarray, voxelspacing=None, connectivity=1
):
    result, reference = np.atleast_1d(result.astype(np.bool_)), np.atleast_1d(
        reference.astype(np.bool_)
    )
    footprint = generate_binary_structure(result.ndim, connectivity)

    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(
        reference, structure=footprint, iterations=1
    )

    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def assd(
    result: torch.Tensor, reference: torch.Tensor, voxelspacing=None, connectivity=1
):
    result, reference = result.cpu().numpy(), reference.cpu().numpy()

    assd = 0.0
    cnt = 0
    for b in range(len(result)):
        _result, _reference = result[b], reference[b]
        if np.sum(_result) != 0 and np.sum(_reference) != 0:
            assd += np.mean(
                (
                    surface_distances(
                        _result, _reference, voxelspacing, connectivity
                    ).mean(),
                    surface_distances(
                        _reference, _result, voxelspacing, connectivity
                    ).mean(),
                )
            )
            cnt += 1
    return assd, cnt


def hd95(
    result: torch.Tensor, reference: torch.Tensor, voxelspacing=None, connectivity=1
):
    result, reference = result.cpu().numpy(), reference.cpu().numpy()
    hd95 = 0.0
    cnt = 0
    for b in range(len(result)):
        _result, _reference = result[b], reference[b]
        if np.sum(_result) != 0 and np.sum(_reference) != 0:
            hd1 = surface_distances(_result, _reference, voxelspacing, connectivity)
            hd2 = surface_distances(_reference, _result, voxelspacing, connectivity)
            hd95 += np.percentile(np.hstack((hd1, hd2)), 95)
            cnt += 1
    return hd95, cnt

import pytest
from SIEDD.configs import (
    MLPConfig,
    NoPosEncode,
    StrainerConfig,
    TrainerConfig,
    EncoderConfig,
    RunConfig,
    QuantizationConfig,
)
from SIEDD.utils.coord_sampler import get_coord_sampler
import torch
from itertools import product


def get_dummy_output_shape(
    strainer: bool, patch_size: int | None, sample_frac: float = 1.0
):
    if patch_size is not None:
        if strainer:
            return (
                5,
                int(sample_frac * 100 * 100 // (patch_size * patch_size)),
                3,
                patch_size,
                patch_size,
            )
        else:
            return (
                1,
                int(sample_frac * 100 * 100 // (patch_size * patch_size)),
                3,
                patch_size,
                patch_size,
            )
    else:
        if strainer:
            return (5, int(sample_frac * 100 * 100), 3)
        else:
            return (1, int(sample_frac * 100 * 100), 3)


@pytest.mark.parametrize(
    "sampling_type,strainer,coordx,patch_size",
    product(["uniform", "edge", None], [False, True], [False, True], [None, 2]),
)
def test_all(sampling_type, strainer, coordx, patch_size):
    if coordx:
        if patch_size is not None:
            coordinates = [
                torch.arange(100 // patch_size),
                torch.arange(100 // patch_size),
            ]
        else:
            coordinates = [torch.arange(100), torch.arange(100)]
    else:
        if patch_size is not None:
            coordinates = torch.ones((100 * 100 // (patch_size * patch_size), 2))
        else:
            coordinates = torch.ones((100 * 100, 2))
    if strainer:
        images = torch.zeros((5, 100, 100, 3))
        train_cfg = StrainerConfig(
            name="strainer",
            sampling=sampling_type,
            coord_sample_frac=0.25,
        )
    else:
        images = torch.zeros((1, 100, 100, 3))
        train_cfg = TrainerConfig(
            name="trainer",
            sampling=sampling_type,
            coord_sample_frac=0.25,
        )

    cs = get_coord_sampler(
        coordinates,
        cfg=RunConfig(
            encoder_cfg=EncoderConfig(
                patch_size=patch_size,
                net=MLPConfig(num_layers=1, pos_encode_cfg=NoPosEncode()),
            ),
            trainer_cfg=train_cfg,
            quant_cfg=QuantizationConfig(),
        ),
        images=images,
    )

    _, out = next(iter(cs))
    frac = 1
    if sampling_type is not None:
        frac = 0.25
    output_shape = get_dummy_output_shape(strainer, patch_size, sample_frac=frac)
    assert out.shape == torch.Size(output_shape), out.shape

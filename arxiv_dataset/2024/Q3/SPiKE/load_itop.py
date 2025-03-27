"""
Module for loading ITOP dataset with augmentations.
"""

from datasets.itop import ITOP

AUGMENT_TRAIN  = [

    {
        "name": "CenterAug",
        "p_prob": 1.0,
        "p_axes": [True, True, True],
    },
    {
        "name": "RotationAug",
        "p_prob": 1.0,
        "p_axis": 1,
        "p_min_angle": -1.57,
        "p_max_angle": 1.57,
    },
    {
        "name": "MirrorAug",
        "p_prob": 1.0,
        "p_axes": [True, False, False],
    },

]

TARGET_FRAME = "last"

dataset = ITOP(
    root="/data/iballester/datasets/ITOP-CLEAN-GT/SIDE",
    num_points=4096,
    frames_per_clip=5,
    train=True,
    use_valid_only=False,
    aug_list=AUGMENT_TRAIN,
    target_frame=TARGET_FRAME,
)

clip, label, frame_idx = dataset[0]

print(clip)
print(label)
print(frame_idx)

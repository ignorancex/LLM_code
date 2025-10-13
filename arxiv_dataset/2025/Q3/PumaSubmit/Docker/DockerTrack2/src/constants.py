### Size thresholds for nuclei (in pixels), pannuke is less conservative
# These have been optimized for the conic challenge, but can be changed
# to get more small nuclei (e.g. by setting all min_threshs to 0)
MIN_THRESHS = [10, 10, 10,10, 10, 10,10, 10, 10,10]
MAX_THRESHS = [20000, 20000, 20000,20000, 20000, 20000,20000, 20000, 20000,2000]

# Maximal size of holes to remove from a nucleus
MAX_HOLE_SIZE = 128

# Colors for geojson output
COLORS_PUMA = [
    [255, 0, 0],  # tumor
    [0, 127, 255],  # neo
    [255, 179, 102],  # other
]

CLASS_LABELS_PUMA = {
    "cell_lymphocyte": 1,
    "cell_tumor": 2,
    "cell_other": 3,
}

CLASS_LABELS_PUMA10 = [
    "nuclei_endothelium",
    "nuclei_plasma_cell",
    "nuclei_stroma",
    "nuclei_tumor",
    "nuclei_histiocyte",
    "nuclei_apoptosis",
    "nuclei_epithelium",
    "nuclei_melanophage",
    "nuclei_neutrophil",
    "nuclei_lymphocyte",
]

# magnifiation and resolutions for WSI dataloader
LUT_MAGNIFICATION_X = [10, 20, 40, 80]
LUT_MAGNIFICATION_MPP = [0.97, 0.485, 0.2425, 0.124]

CONIC_MPP = 0.5
PUMA_MPP = 0.23

# parameters for test time augmentations, do not change
TTA_AUG_PARAMS = {
    "mirror": [
        {"prob_x": 1, "prob_y": 0, "prob": 1},
        {"prob_x": 0, "prob_y": 1, "prob": 1},
        {"prob_x": 1, "prob_y": 1, "prob": 1},
    ],
    "rotate": [
        {"rot90": True, "prob": 1, "degree1": 0},
        {"rot90": True, "prob": 1, "degree1": 90},
        {"rot90": True, "prob": 1, "degree1": -90},
        {"rot90": True, "prob": 1, "degree1": 180},
        {"rot90": True, "prob": 1, "degree1": -180},
    ]
}

# current valid pre-trained weights to be automatically downloaded and used in HoVer-NeXt
VALID_WEIGHTS = []
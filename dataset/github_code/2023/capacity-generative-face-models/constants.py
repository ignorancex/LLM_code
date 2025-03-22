# ///////////// Copyright 2023 Vishnu Boddeti. All rights reserved. /////////////
# //
# //   Project     : Capacity of Generative Face Models
# //   File        : constants.py
# //   Description : Define constants for capacity estimation
# //
# //   Created On: 08/20/2023
# //   Created By: Vishnu Boddeti <mailto:vishnu@msu.edu> and Gautam Sreekumar
# ////////////////////////////////////////////////////////////////////////////


# Colors for printing to terminal
class bcolors:
    DATASET = '\033[91m'
    FACE_MODEL = '\033[96m'
    REF_DATASET = '\033[92m'
    ENDC = '\033[0m'

# Data/logistics constants, please edit these.
ROOT_DIR = "./"
IMG_DIRS = {}
IMG_DIRS["stylegan3"] = "stylegan3/"
IMG_DIRS["stylegan2-ensem"] = "stylegan2-ensem/"
IMG_DIRS["ldm_celebahq_256"] = "ldm_celebahq_256/"
IMG_DIRS["generated.photos"] = "generated.photos/"
IMG_DIRS["pggan"] = "pggan_celebahq_1024"
IMG_DIRS["dcface"] = "dcface_0.5m/"

# Capacity estimation constants
DATASETS = ["pggan", "stylegan2-ensem", "stylegan3",
            "ldm_celebahq_256", "generated.photos", "dcface"]
FACE_MODELS = ["arcface", "adaface"]
REF_DATASETS = ["lfw", "cfp_fp", "cplfw", "agedb_30", "calfw"]

DATASET_LABELS = [r"PG-GAN", r"StyleGAN2+ADA", r"StyleGAN3",
                  r"LDM", r"Generated Photos", r"DCFace"]
FACE_MODEL_LABELS = [r"ArcFace", r"AdaFace"]
REF_DATASET_LABELS = [r"LFW", r"CFP-FP", r"CPLFW", r"AgeDB", r"CALFW"]

FARS = [0.1, 1, 10]
THRESHOLDS = {}
THRESHOLDS["arcface"] = {}
THRESHOLDS["adaface"] = {}

# FAR @ 0.1%, 1%, 10%
THRESHOLDS["arcface"]["lfw"] = [1.575, 1.713, 1.844]
THRESHOLDS["arcface"]["cfp_fp"] = [1.605, 1.696, 1.825]
THRESHOLDS["arcface"]["cplfw"] = [1.525, 1.691, 1.829]
THRESHOLDS["arcface"]["agedb_30"] = [1.530, 1.627, 1.798]
THRESHOLDS["arcface"]["calfw"] = [1.530, 1.662, 1.816]

THRESHOLDS["adaface"]["lfw"] = [1.610, 1.710, 1.847]
THRESHOLDS["adaface"]["cfp_fp"] = [1.641, 1.726, 1.849]
THRESHOLDS["adaface"]["cplfw"] = [1.550, 1.670, 1.832]
THRESHOLDS["adaface"]["agedb_30"] = [1.550, 1.663, 1.818]
THRESHOLDS["adaface"]["calfw"] = [1.530, 1.666, 1.823]

# extracted features, please edit these.
FEAT_FILES = {}
FEAT_FILES['arcface'] = {}
FEAT_FILES['arcface']['stylegan3'] = "arcface/stylegan3.pkl"
FEAT_FILES['arcface']['stylegan2-ensem'] = "arcface/stylegan2-ensem.pkl"
FEAT_FILES['arcface']['ldm_celebahq_256'] = "arcface/ldm_celebahq_256.pkl"
FEAT_FILES['arcface']['generated.photos'] = "arcface/generated.photos.pkl"
FEAT_FILES['arcface']['pggan'] = "arcface/pggan_celebahq_1024.pkl"
FEAT_FILES['arcface']['dcface'] = "arcface/dcface_0.5m.pkl"

FEAT_FILES['adaface'] = {}
FEAT_FILES['adaface']['stylegan3'] = "adaface/stylegan3.pkl"
FEAT_FILES['adaface']['stylegan2-ensem'] = "adaface/stylegan2-ensem.pkl"
FEAT_FILES['adaface']['ldm_celebahq_256'] = "adaface/ldm_celebahq_256.pkl"
FEAT_FILES['adaface']['generated.photos'] = "adaface/generated.photos.pkl"
FEAT_FILES['adaface']['pggan'] = "adaface/pggan_celebahq_1024.pkl"
FEAT_FILES['adaface']['dcface'] = "adaface/dcface_0.5m.pkl"

# Plotting constants
SAVE_PDF = True
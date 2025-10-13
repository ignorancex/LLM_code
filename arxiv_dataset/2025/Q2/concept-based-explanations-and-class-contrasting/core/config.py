DEVICE = "cuda"
SEED = 42


OUTPUT_PATH = "out/" # replace with your own here, the precomputed data will be saved here

# ------------ for imagenet only -------------------------------------------

PATH_TO_IMAGENET_TRAIN = "/localdata/xai_derma/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/"
PATH_TO_IMAGENET_VAL = "/localdata/xai_derma/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/"
PATH_VAL_SOLUTIONS = "/localdata/xai_derma/imagenet-object-localization-challenge/LOC_val_solution.csv"

# ---------------------------------------------------------------


# ---------------------- for digipath only, models and data are in-house, not available -----------------------

PATH_TO_DIGIPATH_DATASET = "derma2_non_normal.dt"
PATH_TO_DIGIPATH_CACHE_FILES = "/localdata/xai_derma/derma2_non_normal.dt_cache"


UNKNOWN= 0
LABELS = {
    None: {
        "value": 0,
        "map_to": UNKNOWN,
        "mask_order": 0,
        "n_patches": 0,
        "metric": False,
    },
    # Basal Cell Carcinoma
    "BCC": {
        "value": 1,
        "map_to": 1,
        "mask_order": 2,
        "n_patches": 50000,
        "metric": True,
    },
    "BCC, Micronodular": {
        "value": 29,
        "map_to": 1,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": False,
    },
    "BCC, Sklerodermiform": {
        "value": 16,
        "map_to": 2,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    "BCC, Infundibulozystisch": {
        "value": 17,
        "map_to": 1,
        "mask_order": 2,
        "n_patches": 5000,
        "metric": False,
    },
    "BCC, Pinkus": {
        "value": 18,
        "map_to": 1,
        "mask_order": 2,
        "n_patches": 5000,
        "metric": False,
    },
    "BCC, Metatypisch": {
        "value": 19,
        "map_to": 1,
        "mask_order": 2,
        "n_patches": 5000,
        "metric": False,
    },
    "BCC, Keratotisch": {
        "value": 21,
        "map_to": 1,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": False,
    },
    "BCC, Pigmentiert": {
        "value": 20,
        "map_to": 3,
        "mask_order": 3,
        "n_patches": 20000,
        "metric": True,
    },
    # Map to 0
    "Epidermis": {
        "value": 2,
        "map_to": UNKNOWN,
        "mask_order": 1,
        "n_patches": 40000,
        "metric": False,
    },
    "Normal": {
        "value": 3,
        "map_to": UNKNOWN,
        "mask_order": 4,
        "n_patches": 60000,
        "metric": False,
    },
    "Stroma": {
        "value": 4,
        "map_to": UNKNOWN,
        "mask_order": 0,
        "n_patches": 0,
        "metric": False,
    },
    "Granuloma": {
        "value": 11,
        "map_to": UNKNOWN,
        "mask_order": 2,
        "n_patches": 1000,
        "metric": False,
    },
    "Tissue": {
        "value": 30,
        "map_to": UNKNOWN,
        "mask_order": 0,
        "n_patches": 0,
        "metric": False,
    },
    # Histiozytome
    "Histiozytome": {
        "value": 5,
        "map_to": 5,
        "mask_order": 2,
        "n_patches": 30000,
        "metric": True,
    },
    # Squamous Cell Carcinoma
    "SCC, Invasive": {
        "value": 6,
        "map_to": 6,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    "SCC, In-situ": {
        "value": 7,
        "map_to": 7,
        "mask_order": 2,
        "n_patches": 30000,
        "metric": True,
    },
    "SCC non-invasiv": {
        "value": 28,
        "map_to": 7,
        "mask_order": 2,
        "n_patches": 0,
        "metric": False,
    },
    # AK
    "AK": {
        "value": 8,
        "map_to": 8,
        "mask_order": 2,
        "n_patches": 30000,
        "metric": True,
    },
    "AK, Bowenoide": {
        "value": 22,
        "map_to": 8,
        "mask_order": 2,
        "n_patches": 5000,
        "metric": False,
    },
    "AK, Hypertrpoh": {
        "value": 23,
        "map_to": 8,
        "mask_order": 2,
        "n_patches": 100,
        "metric": False,
    },
    # SK
    "SK": {
        "value": 9,
        "map_to": 9,
        "mask_order": 2,
        "n_patches": 40000,
        "metric": True,
    },
    "SK, irrit.": {
        "value": 10,
        "map_to": 10,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    # Narbe
    "Narbe": {
        "value": 12,
        "map_to": 11,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    # Nevus
    "blue_nevus": {
        "value": 13,
        "map_to": 12,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    "junctional_melanocytes": {
        "value": 14,
        "map_to": 13,
        "mask_order": 2,
        "n_patches": 50000,
        "metric": True,
    },
    "dermal_melanocytes": {
        "value": 15,
        "map_to": 14,
        "mask_order": 3,
        "n_patches": 30000,
        "metric": True,
    },
    "melanophages": {
        "value": 26,
        "map_to": UNKNOWN,
        "mask_order": 2,
        "n_patches": 5000,
        "metric": True,
    },
    # AFX
    "AFX": {
        "value": 24,
        "map_to": 15,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    "PDS": {
        "value": 25,
        "map_to": 15,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": False,
    },
    # Merkel Cell Carcinoma
    "MCC": {
        "value": 27,
        "map_to": 17,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    # Angiomas (6 subtypes)
    "Angiom": {
        "value": 31,
        "map_to": 18,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    "Angiom_cavernous_hemangioma": {
        "value": 31,
        "map_to": 18,
        "mask_order": 2,
        "n_patches": 0,
        "metric": False,
    },
    "Angiom_tufted": {
        "value": 31,
        "map_to": 18,
        "mask_order": 2,
        "n_patches": 0,
        "metric": False,
    },
    "Angiom_Venous_lake": {
        "value": 31,
        "map_to": 18,
        "mask_order": 2,
        "n_patches": 0,
        "metric": False,
    },
    "Angiom_Venous_Lake": {
        "value": 31,
        "map_to": 18,
        "mask_order": 2,
        "n_patches": 0,
        "metric": False,
    },
    "Angiom_microvenular": {
        "value": 31,
        "map_to": 18,
        "mask_order": 2,
        "n_patches": 0,
        "metric": False,
    },
    "Angiom_arterio_venous": {
        "value": 32,
        "map_to": 18,
        "mask_order": 2,
        "n_patches": 5000,
        "metric": False,
    },
    "Angiom_Arterio_Venous": {
        "value": 32,
        "map_to": 18,
        "mask_order": 2,
        "n_patches": 0,
        "metric": False,
    },
    "Angiom_lobular_hemangioma": {
        "value": 33,
        "map_to": 18,
        "mask_order": 2,
        "n_patches": 30000,
        "metric": True,
    },
    "Granuloma_Pyog": {
        "value": 33,
        "map_to": 18,
        "mask_order": 2,
        "n_patches": 0,
        "metric": False,
    },
    # Trichilemmalcyst
    "Trichilemmalcyst": {
        "value": 35,
        "map_to": 20,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": True,
    },
    # Melanoma
    "melanoma-in-situ": {
        "value": 36,
        "map_to": 21,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    "Melanoma invasiv": {
        "value": 37,
        "map_to": 22,
        "mask_order": 3,
        "n_patches": 20000,
        "metric": True,
    },
    # Other
    # Fibrom maps to Normal
    "Fibrom": {
        "value": 38,
        "map_to": 23,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": True,
    },
    "Epidermalcyst": {
        "value": 39,
        "map_to": 24,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": True,
    },
    "encapsulated_neuroma": {
        "value": 40,
        "map_to": 25,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": True,
    },
    "Neurofibrom": {
        "value": 41,
        "map_to": 26,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    "Verruca": {
        "value": 42,
        "map_to": 27,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    "Toothcyst": {
        "value": 43,
        "map_to": 28,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    # Adnextumors
    "Poroma": {  # 51 slides
        "value": 44,
        "map_to": 29,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": True,
    },
    "Pilomatrixoma": {  # 59 slides
        "value": 45,
        "map_to": 30,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    "Hidradenom": {  # 9 slides
        "value": 46,
        "map_to": 31,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": True,
    },
    # 'Hidradenoma_papilliferum'
    "Trichoblastoma": {  # 33 slides
        "value": 47,
        "map_to": 32,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": True,
    },
    "cylindroma": {  # 26 slides
        "value": 48,
        "map_to": 33,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    "Spiradenoma": {  # 23 slides (maybe join with cylindroma)
        "value": 49,
        "map_to": 34,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    "clear_cell_acanthoma": {  # 18 slides
        "value": 50,
        "map_to": 35,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": True,
    },
    "talgdruesen_tumor_sebaceum": {  # 28 slides
        "value": 51,
        "map_to": 36,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": True,
    },
    "Talgdruesentumor_Adenom": {  # 39 slides
        "value": 52,
        "map_to": 37,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": True,
    },
    "syringom": {  # 20 slides
        "value": 53,
        "map_to": 38,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": True,
    },
    "Chondroid_syringoma": {  # 27 slides
        "value": 54,
        "map_to": UNKNOWN,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": False,
    },
    "Desmoplastic_Trichoepitheliom": {  # 12 slides
        "value": 55,
        "map_to": 40,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": True,
    },
    "Trichoadenom": {  # 17 slides
        "value": 56,
        "map_to": 41,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": True,
    },
    "Hidrocystoma_apocrine": {
        "value": 57,
        "map_to": 42,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    "Trichilemmom": {
        "value": 58,
        "map_to": 43,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": True,
    },
    "Molluscum": {
        "value": 59,
        "map_to": 44,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    "Lentigo solaris": {
        "value": 60,
        "map_to": 45,
        "mask_order": 2,
        "n_patches": 20000,
        "metric": True,
    },
    "Psoriasis": {
        "value": 61,
        "map_to": 46,
        "mask_order": 2,
        "n_patches": 15000,
        "metric": True,
    },
    "Ekzem": {
        "value": 62,
        "map_to": 47,
        "mask_order": 2,
        "n_patches": 15000,
        "metric": True,
    },
    "Bowen-Ca": {
        "value": 63,
        "map_to": 48,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": True,
    },
    "Spitz Nävus": {
        "value": 64,
        "map_to": 49,
        "mask_order": 2,
        "n_patches": 15000,
        "metric": True,
    },
    "Lipom": {
        "value": 65,
        "map_to": 50,
        "mask_order": 2,
        "n_patches": 10000,
        "metric": True,
    },
    "chondrodermatitis": {
        "value": 66,
        "map_to": UNKNOWN,
        "mask_order": 2,
        "n_patches": 15000,
        "metric": False,
    },
}




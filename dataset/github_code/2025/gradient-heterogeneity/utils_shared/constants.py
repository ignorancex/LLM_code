DATASET_NAMES_VISION = ["flowers102", "fgvc_aircraft"]
DATASET_NAMES_SUPERGLUE = ["cb", "rte", "wic", "boolq"]
DATASET_NAMES_GLUE = ["cola", "sst2", "mrpc"]
DATASET_NAMES_NLP = DATASET_NAMES_SUPERGLUE + DATASET_NAMES_GLUE
MODEL_NAMES_VISION = ["resnet18", "vit-base"]
MODEL_NAMES_NLP = ["roberta-base"]
OPTIMIZER_NAMES = [
    "adam",
    "sgd",
    "signsgd",
    "sgd_momentum",
    "signsgd_momentum",
    "rmsprop",
]
TRAINING_MODES = ["normal"]
SEEDS = [1111, 2222, 3333, 4444, 5555]
SEEDS_FOR_TUNING = [1111]
LR_SCHEDULER_TYPES = ["constant", "cosine", "linear_with_warmup"]

from src.trainer import (
    AIGInferencer,
    CombinedGenerationInferencer,
    DraftKTrainer,
    ImageReFLTrainer,
    ImageReFLTrainerV2,
    Inferencer,
    ProFusionInferencer,
    ReFLTrainer,
)

TRAINER_NAME_TO_CLASS = {
    "DraftK": DraftKTrainer,
    "ReFL": ReFLTrainer,
    "ImageReFL": ImageReFLTrainer,
    "ImageReFLV2": ImageReFLTrainerV2,
}

INFERENCER_NAME_TO_CLASS = {
    "Inference": Inferencer,
    "CombinedGenerationInferencer": CombinedGenerationInferencer,
    "ProFusionInferencer": ProFusionInferencer,
    "AIGInferencer": AIGInferencer,
}

REQUIRES_ORIGINAL_MODEL = [
    "CombinedGenerationInferencer",
    "ProFusionInferencer",
    "AIGInferencer",
]

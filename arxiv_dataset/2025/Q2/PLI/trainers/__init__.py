from trainers.trainer import DiffusionTrainer
from trainers.pretrainer import Pretrainer

TRAINER_CODE_DICT = {
    DiffusionTrainer.code(): DiffusionTrainer,
    Pretrainer.code(): Pretrainer
}

def get_trainer_cls(trainer_code):
    return TRAINER_CODE_DICT[trainer_code]
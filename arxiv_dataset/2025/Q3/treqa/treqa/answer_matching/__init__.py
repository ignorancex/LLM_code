from .am_model import BaseAMModel  # noqa: F401
from .chrf_am import ChrfAM
from .exactmatch_am import ExactmatchAM
from .quip_am import QuipAM
from .prompt_am import PromptAM
from .bertscore_am import BertScoreAM

AM_MODELS_REGISTRY = {
    "chrf": ChrfAM,
    "exact_match": ExactmatchAM,
    "quip": QuipAM,
    "prompt_am": PromptAM,
    "bertscore_am": BertScoreAM,
}

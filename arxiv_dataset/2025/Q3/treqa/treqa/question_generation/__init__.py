import sys

from .prompt_qg import PromptQG
from .prompt_qag import PromptQAG
from .qg_model import BaseQGModel, BaseQAGModel  # noqa

# TODO: add proper registry decorators
QG_MODELS_REGISTRY = {
    "prompt_qg": PromptQG,
    "prompt_qag": PromptQAG,
}

try:
    from .lmqg import LMQG

    QG_MODELS_REGISTRY["lmqg"] = LMQG
except ImportError:
    print(
        "WARNING: LMQG is not installed, so it will not be available.",
        file=sys.stderr,
    )

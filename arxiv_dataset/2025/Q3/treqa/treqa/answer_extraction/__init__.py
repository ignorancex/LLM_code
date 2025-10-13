from .spacy_ae import SpacyAE
from .prompt_ae import PromptAE
from .keybert_ae import KeyBertAE

# TODO: add proper registry decorators
AE_MODELS_REGISTRY = {"spacy": SpacyAE, "prompt_ae": PromptAE, "keyllm": KeyBertAE}

from open_biomed.models.protein.progen.progen import ProGen
from open_biomed.models.text.pretrained_lm import PretrainedLMForTextEncoding

TEXT_ENCODER_REGISTRY = {
    "pretrained_lm": PretrainedLMForTextEncoding,
}
TEXT_DECODER_REGISTRY = {}
MOLECULE_ENCODER_REGISTRY = {}
MOLECULE_DECODER_REGISTRY = {}
PROTEIN_ENCODER_REGISTRY = {}
PROTEIN_DECODER_REGISTRY = {
    "progen": ProGen,
}
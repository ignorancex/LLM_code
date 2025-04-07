from .base import BaseQueryModel

AVAILABLE_LMM = {
    "base": "BaseQueryModel",
    "onevision": "OneVision",
    "azure": "Azure",
    "egogpt": "EgoGPT",
    "egogpt_api": "EgoGPTAPI",
    "ixc2_5": "IXC2_5",
    "internvideo2": "InternVideo2",
    "llava": "LLaVA",
    "qwen2_vl": "Qwen2_VL",
    "oryx": "Oryx",
    "longva": "Longva",
    "llava_next": "LLaVA_NeXT",
}


def import_model(model_name):
    """Dynamically import a specific model based on name."""
    model_name = model_name.lower()
    if "llava" in model_name:
        model_name = "llava"
    if model_name not in AVAILABLE_LMM:
        raise ValueError(f"Model {model_name} not found in AVAILABLE_LMM")
    try:
        # Modified import statement to use the correct module path
        module_path = f"egorag.models.{model_name}"
        module = __import__(module_path, fromlist=[AVAILABLE_LMM[model_name]])
        return getattr(module, AVAILABLE_LMM[model_name])
    except Exception as e:
        raise ImportError(f"Failed to import {model_name}. Error: {e}")

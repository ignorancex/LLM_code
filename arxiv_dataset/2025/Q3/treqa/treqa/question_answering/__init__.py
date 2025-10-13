from .qa_model import BaseQAModel  # noqa: F401
from .prompt_qa import PromptQA
from .unified_qa import UnifiedQA

QA_MODELS_REGISTRY = {"prompt_qa": PromptQA, "unified_qa": UnifiedQA}

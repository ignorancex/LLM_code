from packages.llms.base_model import BaseModel
from packages.llms.ollama_model import OllamaModel
from packages.llms.openai_model import OpenAIModel


def choose_model(model_name: str) -> BaseModel:
    """
    Choose the model based on the model name
    :return: a model object
    """
    if model_name in ["gpt-3.5-turbo", 'gpt-4', 'gpt-4-turbo', 'gpt-4o']:
        model = OpenAIModel()
        model.reconfig({"model": model_name})
        return model
    elif model_name in ["llama3:70b-instruct", "qwen:72b-chat", "gemma:7b-instruct", "qwen:7b-chat",
                        "mistral:7b-instruct", "qwen:32b-chat", "gemma2:27b-instruct-fp16", "codestral",
                        "starcoder2:instruct", "qwen2.5-coder:32b-instruct-fp16"]:
        model = OllamaModel()
        model.reconfig({"model": model_name})
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")

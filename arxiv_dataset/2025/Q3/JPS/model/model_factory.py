from .internvl2_8b_model import Internvl28BModel
from .qwen2vl_7b_model import Qwen2VL7BModel
from .minigpt4_13b_model import Minigpt413BModel

class ModelFactory:
    @staticmethod
    def get_model(config):
        model_name = config['model']['name']
        if model_name  == 'internvl2_8b':
            return Internvl28BModel(config)
        elif model_name == 'qwen2vl_7b':
            return Qwen2VL7BModel(config)
        elif model_name == 'minigpt4_13b':
            return Minigpt413BModel(config)
        else:
            raise ValueError(f"未知的模型名称: {model_name}")
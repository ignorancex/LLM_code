from .internvl2_8b_preprocessor import Internvl28BPreprocessor
from .llava_llama3_preprocessor import LlavaLlama3Preprocessor
from .qwen2vl_7b_preprocessor import Qwen2VL7BPreprocessor
from .minigpt4_13b_preprocessor import Minigpt4_13b_Preprocessor

class PreprocessorFactory:
    @staticmethod
    def get_preprocessor(config):
        model_name = config['model']['name']
        if model_name  == 'internvl2_8b':
            return Internvl28BPreprocessor(config['preprocessing'])
        elif model_name == 'qwen2vl_7b':
            return Qwen2VL7BPreprocessor(config['preprocessing'])
        elif model_name == 'minigpt4_13b':
            return Minigpt4_13b_Preprocessor(config['preprocessing'])
        else:
            raise ValueError(f"未知的模型名称: {model_name}")
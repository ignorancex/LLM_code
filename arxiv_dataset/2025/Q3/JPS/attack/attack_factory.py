from .internvl2_8b_attack import Internvl28BAttack
from .llava_llama3_attack import LlavaLlama3Attack
from .qwen2vl_7b_attack import Qwen2VL7BAttack
from .minigpt4_13b_attack import Minigpt413BAttack

class AttackFactory:
    @staticmethod
    def get_attack(config, pipeline):
        model_name = config['model']['name']
        if model_name  == 'internvl2_8b':
            return  Internvl28BAttack(config, pipeline)
        elif model_name == 'llava_llama3':
            return LlavaLlama3Attack(config, pipeline)
        elif model_name == 'qwen2vl_7b':
            return Qwen2VL7BAttack(config, pipeline)
        elif model_name == 'minigpt4_13b':
            return Minigpt413BAttack(config, pipeline)

        # elif dataset_type == 'DatasetB':
        #     return DatasetBLoader()
        # elif dataset_type == 'DatasetC':
        #     return DatasetCLoader()
        else:
            raise ValueError(f"未知的模型名称: {model_name}")
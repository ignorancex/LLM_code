from .llamaguard3_judge import LlamaGuard3Judge
from .harmbench_judge import HarmbenchJudge
from .gpt4o_mini_judge import GPT4oMiniJudge
from .quality_judge import QualityJudge

class JudgeFactory:
    @staticmethod
    def get_judger(config, model_name):
        if model_name  == 'llamaguard3':
            return LlamaGuard3Judge(config)
        elif model_name == 'harmbench':
            return HarmbenchJudge(config)
        elif model_name == 'gpt4omini':
            return GPT4oMiniJudge(config )
        elif model_name == 'quality':
            return QualityJudge(config)
        # elif dataset_type == 'DatasetC':
        #     return DatasetCLoader()
        else:
            raise ValueError(f"未知的模型名称: {model_name}")
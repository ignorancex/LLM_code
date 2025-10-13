from evaluators.diffusion_evaluator import DiffusionEvaluator
from evaluators.blip2_evaluator import BLIP2Evaluator
from evaluators.pretrainer_evaluator import PretrainerEvaluator

evaluator_code_dict = {
    DiffusionEvaluator.code(): DiffusionEvaluator,
    BLIP2Evaluator.code(): BLIP2Evaluator,
    PretrainerEvaluator.code(): PretrainerEvaluator
}
def get_evaluator_cls(evaluator_code):
    return evaluator_code_dict[evaluator_code]
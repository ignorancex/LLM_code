import torch
from torch import nn as nn
from loggers.abc import LoggingService


class DiffusionTrainer(nn.Module):
    def __init__(self, models, val_loggers, evaluators, *args, **kwargs):
        super().__init__()
        self.models = models
        self.val_logging_service = LoggingService(val_loggers)
        self.evaluators = evaluators

    def run(self) -> dict:
        # self._to_eval_mode() # diffusion model is set in evaluation mode (`model.eval()`) by default.
        all_val_results = {}
        # key example: 'fashionIQ_' + clothing_type, CIRR.code()
        for key, evaluator in self.evaluators.items():
            print('results on ' + key)
            val_results = evaluator.evaluate()
            for i in val_results.items():
                all_val_results[key + '_' + i[0]] = i[1]

        self.val_logging_service.log(all_val_results, step=1, commit=True)

        return self.models

    def _to_eval_mode(self, keys=None):
        # keys can be expanded to support multiple models
        self.models.eval()

    @staticmethod
    def _get_state_dicts():
        """
        if model need self-supervised pretraining, we need to save the state_dict of the encoder,
        now we don't update the pretrained parameters
        """
        pass

    @classmethod
    def code(cls) -> str:
        return 'diffusion_trainer'

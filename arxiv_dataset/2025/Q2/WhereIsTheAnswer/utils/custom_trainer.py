from transformers import Trainer
from transformers.utils import logging, is_peft_available,is_safetensors_available, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.modeling_utils import PreTrainedModel, unwrap_model
import torch
from optimum.bettertransformer import BetterTransformer


TRAINING_ARGS_NAME = "training_args.bin"

if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel

logger = logging.get_logger(__name__)
class CustomTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            if "BetterTransformer" in str(self.model):
                BetterTransformer.reverse(self.model).save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )
            else:                
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
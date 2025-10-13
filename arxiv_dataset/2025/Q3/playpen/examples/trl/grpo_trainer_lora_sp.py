from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry

import trl
from peft import LoraConfig

from playpen import BasePlayPen


class PeftGRPOTrainer(BasePlayPen):
    # Note: We configure the proper chat template for the tokenizer already during model loading in the backend

    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)
        # Initialize training configuration
        self.config = trl.GRPOConfig(
            num_train_epochs=10,
            disable_dropout=True,
            max_prompt_length=None,
            max_completion_length=300,
            output_dir=f"models/sft+lora/{self.learner.get_name()}",
            eval_strategy="epoch"
        )
        self.peft_config = LoraConfig(  # see https://huggingface.co/docs/trl/sft_trainer#training-adapters
            r=16, lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            modules_to_save=["lm_head", "embed_token"],
            task_type="CAUSAL_LM",
        )

    def learn(self, game_registry: GameRegistry):
        # Initialize trainer context
        trainer = trl.GRPOTrainer(
            model=self.learner.model,
            processing_class=self.learner.tokenizer,
            train_dataset=...,
            eval_dataset=...,
            args=self.config,
            peft_config=self.peft_config
        )

        # Train on the dataset; this will save only the adapters to the checkpoints directory
        trainer.train()

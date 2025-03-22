import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class Expert:
    def __init__(self, model_id, model_kwargs={}, tokenizer_kwargs={}):
        """
        Initialize the expert model.

        Args:
            model_id (str): The model identifier
            model_kwargs (dict): Keyword arguments for the model
            tokenizer_kwargs (dict): Keyword arguments for the tokenizer
        """
        self.model_id = model_id
        self.model = self._load_model(model_kwargs)
        self.tokenizer = self._load_tokenizer(tokenizer_kwargs)
        self.config = self._load_config()

    def get_name(self):
        """Get the name of the model."""
        return self.model_id.split("/")[-1]

    def _load_model(self, model_kwargs):
        """Load the model."""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )
        return model

    def _load_tokenizer(self, tokenizer_kwargs):
        """Load the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)
        return tokenizer

    def _load_config(self):
        """Load the configuration."""
        config = AutoConfig.from_pretrained(self.model_id)
        return config

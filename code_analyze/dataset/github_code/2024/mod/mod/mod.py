import math
import warnings
import torch

from typing import List
from .expert import Expert
from .config import Configuration


class MoD:
    def __init__(self, config: Configuration):
        self.config = config

    def _load_experts(self):
        """Load the expert models."""
        self.experts = [
            Expert(model_id=e.source_model, model_kwargs=self.config.model_kwargs)
            for e in self.config.experts
        ]

    def _load_weights(self):
        """Load the weights."""
        self.weights = self.config.weights

    def _validate_inputs(self):
        """
        Validate the inputs

        Args:
            experts (List[Expert]): The expert models
            weights (List[float]): The weights
        """
        if len(self.experts) != len(self.weights):
            raise ValueError("The number of experts and weights must be equal")

        if not math.isclose(sum(self.weights), 1.0, rel_tol=1e-5):
            raise ValueError("Weights must sum to 1")

    def _normalize_distribution(self, tensor):
        """
        Normalize a tensor to have values between 0 and 1

        Args:
            tensor (torch.Tensor): The tensor to normalize
        """
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)

        if min_val == max_val:
            return torch.zeros_like(tensor)

        normalized_tensor = (tensor - min_val) / (max_val - min_val)

        return normalized_tensor

    def mixture_of_distributions(self, distributions, weights):
        """
        Mixes a list of distributions based on a list of weights.

        Args:
            distributions (List[torch.Tensor]): A list of distributions to mix
            weights (List[float]): A list of weights to use for the mixture

        Returns:
            torch.Tensor: The mixed distribution
        """
        assert len(distributions) == 2, "Currently only support for two models!"
        # Normalize the distributions
        normalized_dists = [
            self._normalize_distribution(dist) for dist in distributions
        ]

        # Calculate the cumulative weights
        mask = normalized_dists[0] < weights[0]
        mixed_distributions = torch.where(mask, distributions[0], distributions[1])

        return mixed_distributions

    def merge(self, output_path):
        """
        Merge the expert models based on the provided weights

        Returns:
            torch.nn.Module: The merged model
        """

        # Load the experts with weights
        self._load_experts()
        self._load_weights()

        # Validate the inputs
        self._validate_inputs()

        # Load the models
        models = [expert.model for expert in self.experts]

        # Intialize the merged model as the first expert - base model
        merged_model = models[0]

        with torch.no_grad():
            for name, param in merged_model.named_parameters():
                if "mlp" in name:
                    try:
                        distributions = [
                            model.state_dict()[name].data for model in models
                        ]
                        param.data = self.mixture_of_distributions(
                            distributions, self.weights
                        )
                    except Exception as e:
                        warnings.warn(f"Merging layer {name}: {str(e)}")
                        param.data = models[0].state_dict()[name].data

        # Save the merged model
        merged_model.save_pretrained(output_path)
        print(f"Model saved to {output_path}")

        # Save tokenizer and config
        self.experts[0].tokenizer.save_pretrained(output_path)
        print(f"Tokenizer saved to {output_path}")

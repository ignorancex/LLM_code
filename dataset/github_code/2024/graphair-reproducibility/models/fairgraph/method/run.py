from typing import Dict, Optional

import torch

from .graphair import graphair
from .lpgraphair import graphair as lp_graphair
from .fairadj import fairadj
from .fairdrop import fairdrop


class run:
    """
    This class instantiates a fair graph model and implements method to train and evaluate.
    """

    def __init__(self):
        pass

    def run(
        self,
        model: str,
        dataset,
        device,
        config,
        logger,
    ) -> Optional[Dict[str, float]]:
        """This method runs training and evaluation for a fairgraph model on the given dataset.
        Check the README for more information on the models and datasets supported.

        Args:
            model (str): Name of the model to be trained.
            dataset (Dataset): Dataset object containing the dataset to be used.
            device (torch.device): Device to run the model on.
            config (dict): Dictionary containing hyperparameters for the model.
            logger (wandb.run): Wandb logger object to log metrics.
        """

        # Train script
        model_name = model

        model_classes = {
            "graphair": graphair,
            "lpgraphair": lp_graphair,
            "fairadj": fairadj,
            "fairdrop": fairdrop,
        }
        
        selected_model = model_classes[model_name]
        model = selected_model(dataset, config, device, logger)

        # if model has nn.Module class, move it to device
        if isinstance(model, torch.nn.Module):
            model.to(device)

        # train the model
        model.fit()

        # fairdrop and fairadj does not use test functions, so we skip over them
        if hasattr(model, "test"):
            test_results = model.test()
            return test_results

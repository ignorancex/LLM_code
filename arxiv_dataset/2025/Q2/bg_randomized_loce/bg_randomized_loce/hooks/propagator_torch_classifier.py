from typing import Iterable, Dict, Tuple, Union

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .hooks import ForwardHook, ForwardInsertHook, get_module_layer_by_name
from .propagator import Propagator
from ..data_structures.datasets import AbstractDataset
from ..utils.logging import log_error, log_assert


class PropagatorTorchClassifier(Propagator):

    """
    Propagates input tensors through the model and registers activations and/or gradients using ForwardHook and/or BackwardHook
    """

    def __init__(self,
                 model: nn.Module,
                 layers: Union[str, Iterable[str]],
                 batch_size: int = 64,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ) -> None:
        """
        Args:
            model: model
            layers: list of layers for activations and/or gradients registration
            batch_size: butch size for sampling from AbstractDataset instances

        Kwargs:
            batch_size: batch size for conversion of AbstractDataset to DataLoader, default value is 32
            device: torch device
        """
        self.device = device
        self.model = model.to(self.device)
        self.batch_size = batch_size

        self.set_layers(layers)
        
    def set_layers(self, layers: Union[str, Iterable[str]]) -> None:
        """
        Args:
            layers: list of layers for activations and/or gradients registration
        """
        if isinstance(layers, str):
            layers = [layers]

        self.layers = layers
        self.modules = {layer: get_module_layer_by_name(
            self.model, layer) for layer in self.layers}

    def get_predictions(self,
                        input: Union[Tensor, DataLoader, AbstractDataset]
                        ) -> Tensor:
        """
        Propagates the input through the network to get predictions

        Args:
            input: input batch tensor or dataloader or dataset

        Returns:
            predictions: Tensor[...]
        """
        if isinstance(input, Tensor):
            return self.tensor_get_predictions(input)

        elif isinstance(input, DataLoader):
            return self.dataloader_get_predictions(input)

        elif isinstance(input, AbstractDataset):
            return self.dataloader_get_predictions(DataLoader(input, self.batch_size))

        else:
            err_msg = f"wrong type of 'input', current type: {type(input)}"
            log_error(TypeError, err_msg)

    def tensor_get_predictions(self,
                               input: Tensor
                               ) -> Tensor:
        """
        Propagates the input through the network to get predictions

        Args:
            input: input batch tensor

        Returns:
            predictions: Tensor[...]
        """
        with torch.no_grad():
            self.model.eval()

            device_input = input.to(self.device)

            pred = self.model(device_input)

            return pred.detach().cpu()

    def dataloader_get_predictions(self,
                                   input: DataLoader
                                   ) -> Tensor:
        """
        Propagates the input through the network to get predictions

        Args:
            input: input dataloader

        Returns:
            predictions: Tensor[...]
        """

        temp = []

        for batch in tqdm(input, 'Predictions'):

            err_msg = f"'input' dataloader must return tensors, current type: {type(batch)}"
            log_assert(isinstance(batch, Tensor), err_msg)

            temp.append(self.tensor_get_predictions(batch))

        return torch.vstack(temp)

    def get_activations(self,
                        input: Union[Tensor, DataLoader, AbstractDataset]
                        ) -> Dict[str, Tensor]:
        """
        Propagate forward and get activations

        Args:
            input: input batch tensor or dataloader or dataset

        Returns:
            dictionary - {layer: activations}: Dict[str, Tensor[...]]
        """
        if isinstance(input, Tensor):
            return self.tensor_get_activations(input)

        elif isinstance(input, DataLoader):
            return self.dataloader_get_activations(input)

        elif isinstance(input, AbstractDataset):
            return self.dataloader_get_activations(DataLoader(input, self.batch_size))

        else:
            err_msg = f"wrong type of 'input', current type: {type(input)}"
            log_error(TypeError, err_msg)

    def tensor_get_activations(self,
                               input: Tensor
                               ) -> Dict[str, Tensor]:
        """
        Propagate forward and get activations

        Args:
            input: input batch tensor

        Returns:
            dictionary - {layer: activations}: Dict[str, Tensor[...]]
        """
        with torch.no_grad():
            self.model.eval()

            fhooks = [ForwardHook(l, m) for l, m in self.modules.items()]

            device_input = input.to(self.device)

            _ = self.model(device_input)

            activations = {fh.layer_name: fh.get_stacked_activtion()
                        for fh in fhooks}

            fhooks = [fh.remove() for fh in fhooks]

            return activations

    def dataloader_get_activations(self,
                                   input: DataLoader
                                   ) -> Dict[str, Tensor]:
        """
        Propagate forward and get activations

        Args:
            input: input dataloader

        Returns:
            dictionary - {layer: activations}: Dict[str, Tensor[...]]
        """

        temp = []

        for batch in tqdm(input, 'Activations'):

            err_msg = f"'input' dataloader must return tensors, current type: {type(batch)}"
            log_assert(isinstance(batch, Tensor), err_msg)

            temp.append(self.tensor_get_activations(batch))

        acts = {}

        for layer in self.layers:
            acts[layer] = torch.vstack([act[layer] for act in temp])

        return acts

    def get_predictions_from_activations(self,
                                         activations: Union[Tensor, DataLoader, AbstractDataset],
                                         model_input_shape: Tuple[int, int, int],
                                         layer: str = None,
                                         ) -> Tensor:
        """
        Propagates the intermediate activations forward starting from :layer: X to get predictions. Dummy input is used until layer X.

        Args:
            activations: input activations tensor batch or dataloader
            model_input_shape: input shape of model input (without batch dimension: e.g., (3, 224, 224)), required for dummy data pass - Tuple[int, int, int]
        
        Kwargs:
            layer: layer to inject activations and propagate further, if None - self.layers[0]

        Return:
            predictions: Tensor[...]
        """
        if isinstance(activations, Tensor):
            return self.tensor_get_predictions_from_activations(activations, model_input_shape, layer)

        elif isinstance(activations, DataLoader):
            return self.dataloader_get_predictions_from_activations(activations, model_input_shape, layer)

        elif isinstance(activations, AbstractDataset):
            return self.dataloader_get_predictions_from_activations(DataLoader(activations, self.batch_size), model_input_shape, layer)

        else:
            err_msg = f"wrong type of 'activations', current type: {type(activations)}"
            log_error(TypeError, err_msg)

    def tensor_get_predictions_from_activations(self,
                                                input: Tensor,
                                                model_input_shape: Tuple[int, int],
                                                layer: str = None,
                                                ) -> Tensor:
        """
        Propagates the intermediate activations forward starting from :layer: to get predictions. Dummy input is used until layer X.

        Args:
            input: input tensor (B, C, H, W)
            model_input_shape: input shape of model input (without batch dimension: e.g., (3, 224, 224)), required for dummy data pass - Tuple[int, int, int]
        
        Kwargs:
            layer: layer to inject activations and propagate further, if None - self.layers[0]

        Return:
            predictions: Tensor[...]
        """
        with torch.no_grad():
            if layer is None:
                layer = self.layers[0]

            self.model.eval()

            hook = ForwardInsertHook(layer, self.modules[layer])
            device_input = input.to(self.device)
            hook.set_insert_tensor(device_input)  # set injection data

            # batch dimension is added to :model_input_shape:
            batch_shape = (len(device_input), ) + model_input_shape
            # dummy data is used until the injection at layer X
            dummy = torch.zeros(batch_shape)

            pred = self.model(dummy.to(self.device))

            hook.remove()

            return pred.detach().cpu()

    def dataloader_get_predictions_from_activations(self,
                                                    input: DataLoader,
                                                    model_input_shape: Tuple[int],
                                                    layer: str = None,
                                                    ) -> Tensor:
        """
        Propagates the intermediate activations forward starting from :layer: X to get predictions. Dummy input is used until layer X.

        Args:
            input: input dataloader of tensors (B, C, H, W)
            model_input_shape: input shape of model input (without batch dimension: e.g., (3, 224, 224)), required for dummy data pass - Tuple[int, int, int]
        
        Kwargs:
            layer: layer to inject activations and propagate further, if None - self.layers[0]

        Return:
            predictions: Tensor[...]
        """
        if layer is None:
            layer = self.layers[0]

        temp = []

        for batch in tqdm(input, 'Predictions from Activations'):

            err_msg = f"'input' dataloader must return tensors, current type: {type(batch)}"
            log_assert(isinstance(batch, Tensor), err_msg)

            temp.append(self.tensor_get_predictions_from_activations(
                batch, model_input_shape, layer))

        return torch.vstack(temp)

    def get_activations_and_predictions(self,
                                        input: Union[Tensor, DataLoader, AbstractDataset]
                                        ) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Propagate input forward to get activations and predictions

        Args:
            input: input tensor batch or dataloader or dataset

        Returns:
            dictionary {layer:activations}: Dict[str, Tensor[...]]
            predictions: Tensor[...]
        """
        if isinstance(input, Tensor):
            return self.tensor_get_activations_and_predictions(input)

        elif isinstance(input, DataLoader):
            return self.dataloader_get_activations_and_predictions(input)

        elif isinstance(input, AbstractDataset):
            return self.dataloader_get_activations_and_predictions(DataLoader(input, self.batch_size))

        else:
            err_msg = f"wrong type of 'input', current type: {type(input)}"
            log_error(TypeError, err_msg)

    def tensor_get_activations_and_predictions(self,
                                               input: Tensor
                                               ) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Propagate input forward to get activations and predictions

        Args:
            input: input tensor batch

        Returns:
            dictionary {layer:activations}: Dict[str, Tensor[...]]
            predictions: Tensor[...]
        """
        with torch.no_grad():
            self.model.eval()

            fhooks = [ForwardHook(l, m) for l, m in self.modules.items()]

            device_input = input.to(self.device)

            predictions = self.model(device_input)

            activations = {fh.layer_name: fh.get_stacked_activtion()
                        for fh in fhooks}

            fhooks = [fh.remove() for fh in fhooks]

            return activations, predictions.detach().cpu()

    def dataloader_get_activations_and_predictions(self,
                                                   input: DataLoader
                                                   ) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Propagate input forward to get activations and predictions

        Args:
            input: input ataloader

        Returns:
            dictionary {layer:activations}: Dict[str, Tensor[...]]
            predictions: Tensor[...]
        """

        temp_acts = []
        temp_preds = []

        for batch in tqdm(input, 'Activations and Predictions'):

            err_msg = f"'input' dataloader must return tensors, current type: {type(batch)}"
            log_assert(isinstance(batch, Tensor), err_msg)

            a, p = self.tensor_get_activations_and_predictions(batch)
            temp_acts.append(a)
            temp_preds.append(p)

        acts = {}

        for layer in self.layers:
            acts[layer] = torch.vstack([act[layer] for act in temp_acts])

        preds = torch.vstack(temp_preds)

        return acts, preds

from darts.models import NHiTSModel, RNNModel, TFTModel, TCNModel, TransformerModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.utils.likelihood_models import GaussianLikelihood
from utils.ModelTrainArgs import ModelTrainArgs
import torch
from contextlib import contextmanager

@contextmanager
def patch_tensor_deepcopy():
    original_tensor_deepcopy = torch.Tensor.__deepcopy__

    def custom_tensor_deepcopy(self, memo):
        return self.detach().clone()

    torch.Tensor.__deepcopy__ = custom_tensor_deepcopy

    try:
        yield
    finally:
        torch.Tensor.__deepcopy__ = original_tensor_deepcopy


def build_model(model_args : ModelTrainArgs, model_type : str, model_output_dir : str) -> TorchForecastingModel:
    """Builds and returns a TorchForecastingModel based on the provided arguments.

    Parameters
    ----------
    args : list
        List of arguments for configuring the model and training.
    model_output_dir : str
        Path to save the model output.

    Returns
    -------
    TorchForecastingModel
        The built TorchForecastingModel.

    Raises
    ------
    ValueError
        If the provided model type is not supported.
    """
    # Shared model arguments
    shared_model_args = model_args.get_shared_model_args(model_output_dir)
    
    # Model specific arguments
    model_specific_args = model_args.get_model_specific_args()
    
    with patch_tensor_deepcopy():
        match model_type:
            case 'nhits':
                return NHiTSModel(
                    input_chunk_length = model_args.encoder_length,
                    output_chunk_length = model_args.prediction_length,
                    **shared_model_args,
                    **model_specific_args
                )
            case 'lstm':
                # Be aware LSTM does not take output_chunk_length as an argument, it is fixed to 1
                # For longer forecasting horizons just predict multiple times
                return RNNModel(
                    model = 'LSTM',
                    training_length=model_args.encoder_length + model_args.prediction_length,
                    input_chunk_length=model_args.encoder_length,
                    **shared_model_args,
                    **model_specific_args
                )
            case 'deepar':
                return RNNModel(
                    model = 'LSTM',
                    training_length=model_args.encoder_length + model_args.prediction_length,
                    input_chunk_length=model_args.encoder_length,
                    **shared_model_args,
                    **model_specific_args,
                    likelihood=GaussianLikelihood()
                )
            case 'tft':
                return TFTModel(
                    input_chunk_length=model_args.encoder_length,
                    output_chunk_length=model_args.prediction_length,
                    **shared_model_args,
                    **model_specific_args
                )
            case 'tcn':
                return TCNModel(
                    input_chunk_length=model_args.encoder_length,
                    output_chunk_length=model_args.prediction_length,
                    **shared_model_args,
                    **model_specific_args
                )
            case 'transformer':
                return TransformerModel(
                    input_chunk_length=model_args.encoder_length,
                    output_chunk_length=model_args.prediction_length,
                    **shared_model_args,
                    **model_specific_args
                )
            case _:
                raise ValueError(f"Model type {model_type} not supported")
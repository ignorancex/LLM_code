from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.models import NHiTSModel, RNNModel, TFTModel, TCNModel, TransformerModel

def load_model(model_type : str, model_save_path : str) -> TorchForecastingModel:
    """Loads and returns a TorchForecastingModel from the provided save path.

    Parameters
    ----------
    model_type : str
        Type of the model to load.
    model_save_path : str
        Path where the model is saved.

    Returns
    -------
    TorchForecastingModel
        The loaded TorchForecastingModel.

    Raises
    ------
    ValueError
        If the provided model type is not supported.
    """
    
    match model_type:
        case 'nhits':
            return NHiTSModel.load(model_save_path)
        case 'lstm':
            return RNNModel.load(model_save_path)
        case 'deepar':
            return RNNModel.load(model_save_path)
        case 'tft':
            return TFTModel.load(model_save_path)
        case 'tcn':
            return TCNModel.load(model_save_path)
        case 'transformer':
            return TransformerModel.load(model_save_path)
        case _:
            raise ValueError(f"Model type {model_type} is not supported.")
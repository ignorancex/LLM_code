import rootutils
import torch.nn as nn

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.modules import MLP, FeatureMapExtractorModel, RNNHead


class MLP_LSTM_FE(nn.Module):
    def __init__(self, mlp: MLP, decoder: RNNHead, fe: FeatureMapExtractorModel):
        """
        Initializes the MLP_LSTM_FE module.

        Args:
            mlp (MLP): MLP module.
            decoder (RNNHead): Decoder module.
            fe (FeatureMapExtractorModel): FeatureMapExtractorModel module.
        """

        super().__init__()
        self.fe = fe
        self.mlp = mlp
        self.decoder = decoder

    def forward(self, x):
        """
        Performs a forward pass through the MLP_LSTM_FE module.

        Args:
            x (torch.Tensor): Input tensor to be processed.

        Returns:
            torch.Tensor: Output tensor after processing through feature extractor, MLP, and decoder.
        """
        x = self.fe(x)
        x = self.mlp(x)
        x = self.decoder(x)

        return x

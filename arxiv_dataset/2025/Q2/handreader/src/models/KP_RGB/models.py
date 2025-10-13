import rootutils
import torch
import torch.nn as nn

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.modules import MLP, FeatureExtracter, FeatureMapExtractorModel, RNNHead


class TSM_Resnet_Encoder(nn.Module):
    """
    A temporal shift module (TSM) encoder using a ResNet backbone for feature extraction.

    This encoder combines the spatial feature extraction capabilities of ResNet
    with a temporal shift mechanism, allowing for efficient modeling of temporal
    dependencies in video or sequence data.

    Parameters
    ----------
    encoder : str
        The type of ResNet to use as the backbone (e.g., "resnet18", "resnet50").

    unidirection : bool, optional
        If True, the temporal modeling is unidirectional. If False, bidirectional
        modeling is used. Default is False.

    Methods
    -------
    forward(x, input_lengths)
        Computes the forward pass of the encoder.

    """

    def __init__(
        self,
        encoder: str,
        unidirection: bool = False,
    ):
        super().__init__()

        self.backbone = FeatureExtracter(resnet=encoder, unidirection=unidirection)

    def forward(self, x, input_lenghts):
        x = self.backbone(x, input_lenghts)
        return x


class MLP_FE(nn.Module):
    """
    A composite model that combines a feature extractor with a multilayer perceptron (MLP).

    This class integrates a `FeatureMapExtractorModel` for feature extraction and an
    `MLP` for further processing of the extracted features.

    Parameters
    ----------
    mlp : MLP
        An instance of the `MLP` class, used for processing features after extraction.

    fe : FeatureMapExtractorModel
        An instance of the `FeatureMapExtractorModel` class, used for extracting features from input data.

    Methods
    -------
    forward(x, mask=None)
        Processes the input data through the feature extractor and MLP.

    """

    def __init__(self, mlp: MLP, fe: FeatureMapExtractorModel):
        """
        Initializes the MLP_FE module.

        Parameters
        ----------
        mlp : MLP
            An instance of the MLP class, used for processing features after extraction.
        fe : FeatureMapExtractorModel
            An instance of the FeatureMapExtractorModel class, used for extracting features from input data.
        """
        super().__init__()
        self.fe = fe
        self.mlp = mlp

    def forward(self, x):
        """
        Processes the input data through the feature extractor and the multilayer perceptron.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to be processed by the feature extractor and MLP.

        Returns
        -------
        torch.Tensor
            The output tensor after processing by the feature extractor and MLP.
        """
        x = self.fe(x)
        x = self.mlp(x)

        return x


class JointEncoders(nn.Module):
    """
    A module that integrates multiple encoders and a decoder to process RGB and keypoint data.

    This class combines an RGB encoder (`TSM_Resnet_Encoder`), a keypoint encoder (`MLP_FE`),
    and a decoder (`RNNHead`). It allows different operations like sum, concatenation, product,
    or weighted combinations of the encoder outputs based on the specified reduction method.

    Parameters
    ----------
    encoder_rgb : TSM_Resnet_Encoder
        An encoder for processing RGB data.

    encoder_kp : MLP_FE
        An encoder for processing keypoint data.

    decoder : nn.Module
        The decoder module used to process the encoded features.

    decoder_net : RNNHead
        The RNN-based decoder network.

    reduction : str
        Specifies the method of combining the outputs of `encoder_rgb` and `encoder_kp`.
        Options are: "sum", "concat", "prod", "weight_sum", "weight_sum2".

    Attributes
    ----------
    encoder_rgb : TSM_Resnet_Encoder
        Stores the RGB encoder.

    encoder_kp : MLP_FE
        Stores the keypoint encoder.

    decoder : nn.Module
        Stores the decoder module.

    decoder_net : RNNHead
        Stores the RNN-based decoder network.

    reduction : str
        Specifies how the encoded outputs are combined.

    weights1 : nn.Parameter
        Parameter used for weighted sum reduction.

    weights2 : nn.Parameter
        Parameter used for weighted sum2 reduction.

    Methods
    -------
    forward(x_rgb, x_kp, input_lenghts=None)
        Processes the input tensors through the encoders, applies the specified reduction, and then
        passes the combined output through the decoder network.

    """

    def __init__(
        self,
        encoder_rgb: TSM_Resnet_Encoder,
        encoder_kp: MLP_FE,
        decoder,
        decoder_net: RNNHead,
        reduction: str,
    ):
        """
        Initializes the JointEncoders module.

        Parameters
        ----------
        encoder_rgb : TSM_Resnet_Encoder
            An encoder for processing RGB data.

        encoder_kp : MLP_FE
            An encoder for processing keypoint data.

        decoder : nn.Module
            The decoder module used to process the encoded features.

        decoder_net : RNNHead
            The RNN-based decoder network.

        reduction : str
            Specifies the method of combining the outputs of `encoder_rgb` and `encoder_kp`.
            Options are: "sum", "concat", "prod", "weight_sum", "weight_sum2".

        Attributes
        ----------
        encoder_rgb : TSM_Resnet_Encoder
            Stores the RGB encoder.

        encoder_kp : MLP_FE
            Stores the keypoint encoder.

        decoder : nn.Module
            Stores the decoder module.

        decoder_net : RNNHead
            Stores the RNN-based decoder network.

        reduction : str
            Specifies how the encoded outputs are combined.

        weights1 : nn.Parameter
            Parameter used for weighted sum reduction.

        weights2 : nn.Parameter
            Parameter used for weighted sum2 reduction.
        """
        super().__init__()
        self.encoder_rgb = encoder_rgb
        self.encoder_kp = encoder_kp
        self.decoder = decoder
        self.decoder_net = decoder_net
        self.reduction = reduction
        self.weights1 = nn.Parameter(torch.randn(4, 1, 1024)).cuda()
        self.weights2 = nn.Parameter(torch.randn(4, 1, 1024)).cuda()

    def forward(self, x_rgb, x_kp, input_lenghts=None):
        """
        Forward pass through the joint encoders.

        Parameters
        ----------
        x_rgb : torch.Tensor
            Input RGB features.

        x_kp : torch.Tensor
            Input keypoint features.

        input_lenghts : list, optional
            List of sequence lengths for each sample in the batch.

        Returns
        -------
        torch.Tensor
            Output features after encoding and reduction.
        """
        if input_lenghts is not None:
            x_rgb = self.encoder_rgb(x_rgb, input_lenghts)
        else:
            x_rgb = self.encoder_rgb(x_rgb)

        x_kp = self.encoder_kp(x_kp)

        if self.reduction == "sum":
            x = x_kp + x_rgb
        elif self.reduction == "concat":
            x = torch.cat((x_rgb, x_kp), -1)
        elif self.reduction == "prod":
            x = x_rgb * x_kp
        elif self.reduction == "weight_sum":
            x = x_rgb * self.weights1 + x_kp * self.weights1
        elif self.reduction == "weight_sum2":
            x = x_rgb * self.weights1 + x_kp * self.weights2
        else:
            raise Exception("wrong reduction")

        x = self.decoder_net(x)

        return x

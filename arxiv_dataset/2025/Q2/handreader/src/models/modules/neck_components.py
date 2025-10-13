import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self) -> None:
        """
        Initializes the Swish activation module.
        """
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Applies the Swish activation function to the input tensor.

        The Swish activation function is defined as `x * sigmoid(x)`, which is
        a smooth, non-monotonic function that can improve model performance
        compared to traditional activation functions like ReLU.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor to apply the Swish activation function to.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the Swish activation.
        """
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        """
        Initializes the Gated Linear Unit (GLU) module.

        The Gated Linear Unit is a type of activation function that takes as input a
        tensor of shape `(batch_size, sequence_length, hidden_size)` and outputs a
        tensor of the same shape.

        Parameters
        ----------
        dim : int
            The dimensionality of the output space.
        """
        super().__init__()
        self.dim = dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Applies the Gated Linear Unit (GLU) activation function to the input tensor.

        The Gated Linear Unit is a type of activation function that takes as input a
        tensor of shape `(batch_size, sequence_length, hidden_size)` and outputs a
        tensor of the same shape.

        The GLU activation function is defined as the element-wise product of
        the input tensor and the sigmoid of the input tensor. This can be expressed
        as `x * sigmoid(x)`.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor to apply the GLU activation function to.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the GLU activation.
        """
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    ref : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: False
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        """
        Initializes the DepthwiseConv1d module.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input
        out_channels : int
            Number of channels produced by the convolution
        kernel_size : int
            Size of the convolving kernel
        stride : int, optional
            Stride of the convolution. Default: 1
        padding : int or tuple, optional
            Zero-padding added to both sides of the input. Default: 0
        bias : bool, optional
            If True, adds a learnable bias to the output. Default: False
        """
        super().__init__()
        assert (
            out_channels % in_channels == 0
        ), "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Applies the depthwise 1D convolution to the input.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, in_channels, sequence_length)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, sequence_length)
        """
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    """
    When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution.
    This operation often used to match dimensions.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        """
        Initializes the PointwiseConv1d module.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input
        out_channels : int
            Number of channels produced by the convolution
        stride : int, optional
            Stride of the convolution. Default: 1
        padding : int or tuple, optional
            Zero-padding added to both sides of the input. Default: 0
        bias : bool, optional
            If True, adds a learnable bias to the output. Default: True
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Applies the pointwise 1-D convolution to the input.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, in_channels, sequence_length)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, sequence_length)
        """
        return self.conv(inputs)


class ConvModule(nn.Module):
    """
    Convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout
    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by squeezeformer convolution module.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        """
        Initializes the ConvModule instance.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input
        kernel_size : int or tuple, optional
            Size of the convolving kernel Default: 31
        expansion_factor : int, optional
            Expansion factor to controls width of output. Default: 2
        dropout_p : float, optional
            probability of dropout. Default: 0.1
        """
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.pw_conv_1 = PointwiseConv1d(
            in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True
        )
        self.act1 = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(
            in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2
        )
        self.bn = nn.BatchNorm1d(in_channels)
        self.act2 = Swish()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)

    def forward(self, x):
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        x = x.transpose(1, 2)
        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)

        x = self.bn(x)

        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x)
        x = x.transpose(1, 2)
        return x


class FeatureMapExtractorModel(nn.Module):
    """
    A feature map extraction model that processes input data through convolutional and instance normalization layers.

    Parameters:
    -----------
    num_keypoints : int
        Number of keypoints to consider in the input data.
    out_dim : int
        Dimension of the output feature vectors.
    """

    def __init__(self, num_keypoints: int, out_dim: int):
        """
        Initializes the FeatureMapExtractorModel instance.

        Parameters
        ----------
        num_keypoints : int
            Number of keypoints to consider in the input data.
        out_dim : int
            Dimension of the output feature vectors.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding="same")
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(1, 1, (5, 1, 1), stride=(3, 1, 1))
        self.IN = nn.InstanceNorm2d([10])
        self.relu3 = nn.ReLU()
        num_f = num_keypoints * 5
        self.linear = nn.Linear(num_f, out_dim)

    def forward(self, x):
        """
        Computes the forward pass of the FeatureMapExtractorModel.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (#batch, #keypoints, #frames, 3).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (#batch, #frames, #features).
        """

        x = x.permute(0, 3, 1, 2)
        bs, ch, fr, kp = x.shape
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        # make channel equals 1 for 3dconv
        # [bs, 1, coords, frames, keypoints]
        x = torch.unsqueeze(x, 1)
        # return to [bs, coords, frames, keypoints]
        x = torch.squeeze(self.conv3(x), 1)
        x = self.IN(x)
        # make [bs, coords, frames, keypoints] to [bs, frames, coords, keypoints]
        permuted_x = torch.permute(x, (0, 2, 1, 3))
        x = permuted_x.reshape(bs, fr, -1)
        x = self.linear(x)
        return x


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) module with optional double convolutional layers.

    Parameters:
    -----------
    input_dim : int
        The dimension of the input features.
    hidden_dim : int
        The dimension of the hidden layer.
    output_dim : int
        The dimension of the output layer.
    double_conv : bool, optional
        Whether to include an additional convolutional layer after the initial convolution (default: False).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        double_conv: bool = False,
    ):
        """
        Initializes the MLP instance.

        Parameters
        ----------
        input_dim : int
            The dimension of the input features.
        hidden_dim : int
            The dimension of the hidden layer.
        output_dim : int
            The dimension of the output layer.
        double_conv : bool, optional
            Whether to include an additional convolutional layer after the initial convolution (default is False).
        """
        super().__init__()
        self.double_conv = double_conv

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu2 = nn.ReLU()

        self.convs = ConvModule(in_channels=output_dim)
        if self.double_conv:
            self.convs2 = ConvModule(in_channels=output_dim)

    def forward(self, x):
        """
        Computes the forward pass of the MLP module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (#batch, #frames, #features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (#batch, #frames, #output_dim).
        """
        bs, t, f = x.shape

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.convs(x)
        if self.double_conv:
            x = self.convs2(x)

        return x

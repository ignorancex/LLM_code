import torch
import torch.nn as nn
from .cbam import CBAM


class Scissor(torch.nn.Module):
    # Remove the first row and column of our data
    # To deal with asymmetry in ConvTranpose layers
    # if used correctly, this removes 0's
    def forward(self, x):
        _, _, h, _ = x.shape
        x = x[:,:,1:h,1:h]
        return x


class TaskonomyDecoder(nn.Module):
    """
    Note regarding DeConvolution Layer:
    - TF uses padding = 'same': `o = i * stride` (e.g. 128 -> 64 if stride = 2)
    - Using the equation relating output_size, input_size, stride, padding, kernel_size, we get 2p = 1
    - See https://stackoverflow.com/questions/50683039/conv2d-transpose-output-shape-using-formula
    - This means we need to add asymmetric padding of (1,0,1,0) prior to deconv
    - PyTorch ConvTranspose2d does not support asymmetric padding, so we need to pad ourselves
    - But since we pad ourselves it goes into the input size and since stride = 2, we get an extra row/column of zeros
    - e.g. This is because it is putting a row/col between each row/col of the input (our padding is treated as input)
    - That's fine, if we remove that row and column, we get the proper outputs we are looking for
    - See https://github.com/vdumoulin/conv_arithmetic to visualize deconvs
    """

    def __init__(self, out_channels=3, eval_only=False):
        super(TaskonomyDecoder, self).__init__()
        self.fc = nn.Linear(512, 14 * 14 * 128)  # Adjust this to match the desired shape

        self.conv2 = self._make_layer(128, 128)
        self.conv3 = self._make_layer(128, 128)
        self.conv4 = self._make_layer(128, 64)
        self.conv5 = self._make_layer(64, 64)
        self.conv6 = self._make_layer(64, 32)
        self.conv7 = self._make_layer(32, 32)

        self.deconv8 = self._make_layer(32, 16, stride=2, deconv=True)
        self.conv9 = self._make_layer(16, 16)

        self.deconv10 = self._make_layer(16, 8, stride=2, deconv=True)
        self.conv11 = self._make_layer(8, 8)

        self.deconv12 = self._make_layer(8, 4, stride=2, deconv=True)
        self.conv13 = self._make_layer(4, 4)

        self.deconv14 = self._make_layer(4, 2, stride=2, deconv=True)
        
        if out_channels != 0:
            self.decoder_output = nn.Sequential(
                nn.Conv2d(2, out_channels, kernel_size=3, stride=1, bias=True, padding=1),
                nn.Tanh()
            )
        else:
            self.decoder_output = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2 * 224 * 224, 16)
            )

        self.eval_only = eval_only
        if self.eval_only:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    def _make_layer(self, in_channels, out_channels, stride=1, deconv=False):
        if deconv:
            pad = nn.ZeroPad2d((1,0,1,0))  # Pad first row and column
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=0, bias=False)
            scissor = Scissor()  # Remove first row and column
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)  # pad = 'SAME'

        bn = nn.BatchNorm2d(out_channels, momentum=0.1, affine=True)
        lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        if deconv:
            layer = nn.Sequential(pad, conv, scissor, bn, lrelu)
        else:
            layer = nn.Sequential(conv, bn, lrelu)
        return layer

    def forward(self, x):
        # Reshape the input
        x = self.fc(x)
        x = x.view(x.size(0), 128, 14, 14)  # Adjust this to match the desired shape
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.deconv8(x)
        x = self.conv9(x)

        x = self.deconv10(x)
        x = self.conv11(x)

        x = self.deconv12(x)
        x = self.conv13(x)

        x = self.deconv14(x)
        output = self.decoder_output(x)
        return x, output
    

class TaskonomyDecoderCBAM(nn.Module):
    """
    Note regarding DeConvolution Layer:
    - TF uses padding = 'same': `o = i * stride` (e.g. 128 -> 64 if stride = 2)
    - Using the equation relating output_size, input_size, stride, padding, kernel_size, we get 2p = 1
    - See https://stackoverflow.com/questions/50683039/conv2d-transpose-output-shape-using-formula
    - This means we need to add asymmetric padding of (1,0,1,0) prior to deconv
    - PyTorch ConvTranspose2d does not support asymmetric padding, so we need to pad ourselves
    - But since we pad ourselves it goes into the input size and since stride = 2, we get an extra row/column of zeros
    - e.g. This is because it is putting a row/col between each row/col of the input (our padding is treated as input)
    - That's fine, if we remove that row and column, we get the proper outputs we are looking for
    - See https://github.com/vdumoulin/conv_arithmetic to visualize deconvs
    """

    def __init__(self, out_channels=3, eval_only=False):
        super(TaskonomyDecoderCBAM, self).__init__()
        self.fc = nn.Linear(512, 14 * 14 * 128)  # Adjust this to match the desired shape
        self.cbam_module = CBAM(128, 7).type(torch.float32)
        self.conv2 = self._make_layer(128, 128)
        self.conv3 = self._make_layer(128, 128)
        self.conv4 = self._make_layer(128, 64)
        self.conv5 = self._make_layer(64, 64)
        self.conv6 = self._make_layer(64, 32)
        self.conv7 = self._make_layer(32, 32)

        self.deconv8 = self._make_layer(32, 16, stride=2, deconv=True)
        self.conv9 = self._make_layer(16, 16)

        self.deconv10 = self._make_layer(16, 8, stride=2, deconv=True)
        self.conv11 = self._make_layer(8, 8)

        self.deconv12 = self._make_layer(8, 4, stride=2, deconv=True)
        self.conv13 = self._make_layer(4, 4)

        self.deconv14 = self._make_layer(4, 2, stride=2, deconv=True)
        
        if out_channels != 0:
            self.decoder_output = nn.Sequential(
                nn.Conv2d(2, out_channels, kernel_size=3, stride=1, bias=True, padding=1),
                nn.Tanh()
            )
        else:
            self.decoder_output = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2 * 224 * 224, 16)
            )

        self.eval_only = eval_only
        if self.eval_only:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    def _make_layer(self, in_channels, out_channels, stride=1, deconv=False):
        if deconv:
            pad = nn.ZeroPad2d((1,0,1,0))  # Pad first row and column
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=0, bias=False)
            scissor = Scissor()  # Remove first row and column
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)  # pad = 'SAME'

        bn = nn.BatchNorm2d(out_channels, momentum=0.1, affine=True)
        lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        if deconv:
            layer = nn.Sequential(pad, conv, scissor, bn, lrelu)
        else:
            layer = nn.Sequential(conv, bn, lrelu)
        return layer

    def forward(self, x):
        # Reshape the input
        x = self.fc(x)
        x = x.view(x.size(0), 128, 14, 14)  # Adjust this to match the desired shape
        x = self.cbam_module(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.deconv8(x)
        x = self.conv9(x)

        x = self.deconv10(x)
        x = self.conv11(x)

        x = self.deconv12(x)
        x = self.conv13(x)

        x = self.deconv14(x)
        output = self.decoder_output(x)
        return x, output
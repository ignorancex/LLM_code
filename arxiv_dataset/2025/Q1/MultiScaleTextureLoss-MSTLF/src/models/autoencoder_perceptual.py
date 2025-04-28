import torch
import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU()

        # Decoder layers
        self.conv8_d = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu8_d = nn.ReLU()
        self.conv7_d = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu7_d = nn.ReLU()
        self.conv6_d = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu6_d = nn.ReLU()
        self.conv5_d = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu5_d = nn.ReLU()

        self.upsample2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        self.conv4_d = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu4_d = nn.ReLU()
        self.conv3_d = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu3_d = nn.ReLU()

        self.upsample1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.conv2_d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2_d = nn.ReLU()
        self.conv1_d = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.relu1_d = nn.ReLU()

    """def forward(self, x):
        # Encoder
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool2(x)
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        encoded_tensor = self.relu8(self.conv8(x))

        # Decoder
        x = self.relu8_d(self.conv8_d(encoded_tensor))
        x = self.relu7_d(self.conv7_d(x))
        x = self.relu6_d(self.conv6_d(x))
        x = self.relu5_d(self.conv5_d(x))
        x = self.upsample2(x)
        x = self.relu4_d(self.conv4_d(x))
        x = self.relu3_d(self.conv3_d(x))
        x = self.upsample1(x)
        x = self.relu2_d(self.conv2_d(x))
        decoded_tensor = self.relu1_d(self.conv1_d(x))

        return encoded_tensor, decoded_tensor"""

    # with skip connections
    def forward(self, x):
        # Encoder
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(x1))
        x3 = self.pool1(x2)
        x4 = self.relu3(self.conv3(x3))
        x5 = self.relu4(self.conv4(x4))
        x6 = self.pool2(x5)
        x7 = self.relu5(self.conv5(x6))
        x8 = self.relu6(self.conv6(x7))
        x9 = self.relu7(self.conv7(x8))
        encoded_tensor = self.relu8(self.conv8(x9))

        # Decoder
        x9_d = self.relu8_d(self.conv8_d(encoded_tensor))
        x8_d = self.relu7_d(self.conv7_d(x9_d))
        x7_d = self.relu6_d(self.conv6_d(x8_d))
        x6_d = self.relu5_d(self.conv5_d(x7_d))
        x5_d = self.upsample2(x6_d)
        x5_d = x5_d + x5  # Adding skip connection
        x4_d = self.relu4_d(self.conv4_d(x5_d))
        x3_d = self.relu3_d(self.conv3_d(x4_d))
        x2_d = self.upsample1(x3_d)
        x2_d = x2_d + x2  # Adding skip connection
        x1_d = self.relu2_d(self.conv2_d(x2_d))
        decoded_tensor = self.relu1_d(self.conv1_d(x1_d))

        return encoded_tensor, decoded_tensor


# perceptual loss
def perceptual_autoencoder_loss(x, y, autoencoder):
    encoded_x, _ = autoencoder(x)
    encoded_y, _ = autoencoder(y)
    loss = torch.mean(torch.mean((encoded_x - encoded_y) ** 2))
    return loss


if __name__ == "__main__":
    # Example usage:
    autoencoder = Autoencoder().apply(weights_init_normal)

    input_tensor = torch.randn((1, 1, 256, 256))  # Assuming input image size is 256x256
    encoded_tensor, decoded_tensor = autoencoder(input_tensor)

    print("Encoded Tensor Shape:", encoded_tensor.shape)
    print("Decoded Tensor Shape:", decoded_tensor.shape)

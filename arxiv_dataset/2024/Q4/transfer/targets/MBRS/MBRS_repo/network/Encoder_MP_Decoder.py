from . import *
from .Encoder_MP import Encoder_MP, Encoder_MP_Diffusion
from .Decoder import Decoder, Decoder_Diffusion
from .Noise import Noise
import sys

sys.path.append('/scratch/qilong3/watermark_robustness')
sys.path.append('/scratch/qilong3/transferattack')
from add_noise import add_noise


class EncoderDecoder(nn.Module):
    '''
    A Sequential of Encoder_MP-Noise-Decoder
    '''

    def __init__(self, H, W, message_length, noise_layers, device):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder_MP(H, W, message_length)
        self.noise = Noise(noise_layers)
        self.decoder = Decoder(H, W, message_length)
        self.device = device

    def forward(self, image, message):
        encoded_images = self.encoder(image, message)
        noised_images = add_noise(encoded_images, self.device)
        # noised_image = self.noise([encoded_images, image])
        decoded_message = self.decoder(noised_images)
        return encoded_images, noised_images, decoded_message


class EncoderDecoder_Diffusion(nn.Module):
    '''
    A Sequential of Encoder_MP-Noise-Decoder
    '''

    def __init__(self, H, W, message_length, noise_layers):
        super(EncoderDecoder_Diffusion, self).__init__()
        self.encoder = Encoder_MP_Diffusion(H, W, message_length)
        self.noise = Noise(noise_layers)
        self.decoder = Decoder_Diffusion(H, W, message_length)

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        noised_image = self.noise([encoded_image, image])
        decoded_message = self.decoder(noised_image)

        return encoded_image, noised_image, decoded_message

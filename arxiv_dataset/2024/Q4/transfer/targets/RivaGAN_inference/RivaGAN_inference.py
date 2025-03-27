import torch
from imwatermark import WatermarkEncoder, WatermarkDecoder
import numpy as np


class RivaGAN:
    class Encoder:
        def __init__(self):
            # Initialize the WatermarkEncoder
            self.wm_encoder = WatermarkEncoder()
            # Load the pre-trained RivaGAN model
            WatermarkEncoder.loadModel()

        def eval(self):
            # Do nothing when .eval() is called
            pass

        def __call__(self, images, messages):
            """
            Encode the images with the given watermark messages.

            Parameters:
            - images: A batch of images as Tensors (N, 3, 128, 128) in [-1, 1] range.
            - messages: A list of watermark keys (binary strings) or a single binary string.

            Returns:
            - wm_images: The watermarked images as a Tensor of shape (N, 3, 128, 128) in [-1, 1] range.
            """
            # Ensure images is a batch of images
            if images.ndim == 3:
                images = images.unsqueeze(0)  # Convert single image to batch

            N = images.shape[0]

            # If a single message is provided, repeat it for all images
            if isinstance(messages, str):
                messages = [messages] * N

            # Upscale images to 256x256 in batch
            images = torch.nn.functional.interpolate(
                images, size=(256, 256), mode='bilinear', align_corners=False
            )

            # Convert from [-1, 1] to [0, 255] uint8 in batch
            images_np = ((images.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5).astype(np.uint8).astype(np.float32)
            # round to nearest integer

            wm_images_np = []

            for img_np, wm_key in zip(images_np, messages):
                self.wm_encoder.set_watermark('bits', wm_key)
                # Encode the image using the 'rivaGan' method
                wm_img_np = self.wm_encoder.encode(img_np, 'rivaGan')
                wm_images_np.append(wm_img_np)

            # Convert list of numpy arrays to a single numpy array
            wm_images_np = np.stack(wm_images_np, axis=0)

            # Convert back to [-1, 1] tensor in batch
            wm_images = torch.from_numpy(wm_images_np.astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2)

            # Downscale back to 128x128 in batch
            wm_images = torch.nn.functional.interpolate(
                wm_images, size=(128, 128), mode='bilinear', align_corners=False
            )

            # Move to the same device as input images
            wm_images = wm_images.to(images.device)
            return wm_images

    class Decoder:
        def __init__(self, wm_key_length):
            # Initialize the WatermarkDecoder
            self.wm_decoder = WatermarkDecoder('bits', wm_key_length)
            # Load the pre-trained RivaGAN model
            WatermarkDecoder.loadModel()

        def eval(self):
            # Do nothing when .eval() is called
            pass

        def __call__(self, images):
            """
            Decode the watermarks from the images.

            Parameters:
            - images: A batch of images as Tensors (N, 3, 128, 128) in [-1, 1] range.

            Returns:
            - wm_keys: A list of extracted watermark keys.
            """
            # Ensure images is a batch of images
            if images.ndim == 3:
                images = images.unsqueeze(0)  # Convert single image to batch

            N = images.shape[0]

            # Upscale images to 256x256 in batch
            images = torch.nn.functional.interpolate(
                images, size=(256, 256), mode='bilinear', align_corners=False
            )

            # Convert from [-1, 1] to [0, 255] uint8 in batch
            images_np = ((images.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5).astype(np.uint8).astype(np.float32)

            wm_keys = []

            for img_np in images_np:
                # Decode the watermark using the 'rivaGan' method
                wm_key = self.wm_decoder.decode(img_np, 'rivaGan')
                wm_keys.append(wm_key)

            # Convert list of keys to tensor
            wm_keys = torch.tensor(np.array(wm_keys))
            # Move to the same device as input images
            wm_keys = wm_keys.to(images.device)
            return wm_keys

    def __init__(self, device, wm_key_length=32):
        self.device = device
        # Initialize the Encoder and Decoder classes
        self.encoder = self.Encoder()
        self.decoder = self.Decoder(wm_key_length)
        # Note: WatermarkEncoder and WatermarkDecoder do not have 'to(device)' methods

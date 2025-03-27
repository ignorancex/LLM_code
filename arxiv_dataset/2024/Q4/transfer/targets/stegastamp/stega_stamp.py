import torch
import torch.nn as nn
import numpy as np
import cv2
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants


class StegaStampModel:
    def __init__(self, model_path, device,message_length=30):
        """
        Initializes the StegaStampModel.

        Args:
            model_path (str): Path to the saved StegaStamp TensorFlow model.
            message_length (int): Length of the binary message to embed/extract.
        """
        self.message_length = 100  # Define the fixed message length

        # Configure TensorFlow to use CPU only
        config = tf.ConfigProto(
            device_count={'GPU': 0},
            allow_soft_placement=True,
            log_device_placement=False
        )

        # Initialize TensorFlow session with the CPU-only configuration
        self.sess = tf.InteractiveSession(graph=tf.Graph(), config=config)

        # Load the TensorFlow model
        model = tf.saved_model.loader.load(self.sess, [tag_constants.SERVING], model_path)

        # Get input and output tensors for the encoder
        signature = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.input_secret_name = signature.inputs['secret'].name
        self.input_image_name = signature.inputs['image'].name
        self.input_secret = tf.get_default_graph().get_tensor_by_name(self.input_secret_name)
        self.input_image = tf.get_default_graph().get_tensor_by_name(self.input_image_name)

        self.output_stegastamp_name = signature.outputs['stegastamp'].name
        self.output_residual_name = signature.outputs['residual'].name
        self.output_stegastamp = tf.get_default_graph().get_tensor_by_name(self.output_stegastamp_name)
        self.output_residual = tf.get_default_graph().get_tensor_by_name(self.output_residual_name)

        # Get input and output tensors for the decoder
        self.output_secret_name = signature.outputs['decoded'].name
        self.output_secret = tf.get_default_graph().get_tensor_by_name(self.output_secret_name)

        # Define encoder and decoder as instances of inner classes
        self.encoder = self.Encoder(self)
        self.decoder = self.Decoder(self)
        self.device = device

    class Encoder:
        def __init__(self, parent):
            self.parent = parent

        def eval(self):
            pass  # Placeholder to match PyTorch model interface

        def __call__(self, images, messages):
            """
            Encodes messages into images.

            Args:
                images (torch.Tensor): Batch of images with shape (batch_size, C, H, W).
                messages (torch.Tensor): Batch of binary messages with shape (batch_size, message_length).

            Returns:
                torch.Tensor: Batch of images with embedded messages.
            """
            # Convert torch tensors to numpy arrays
            images_np = images.detach().cpu().numpy()
            messages_np = messages.detach().cpu().numpy()

            # Convert images from [-1, 1] to [0, 1]
            images_np = (images_np + 1) / 2.0

            images_np = images_np.transpose(0, 2, 3, 1)
            # Resize to (400, 400)
            images_np = np.array([cv2.resize(image, (400, 400), interpolation=cv2.INTER_LINEAR) for image in images_np])

            # Ensure messages are binary and of fixed length
            batch_size = images_np.shape[0]
            fixed_length = self.parent.message_length

            # Prepare secrets by padding or truncating messages to fixed_length
            secrets = []
            for message_bits in messages_np:
                bits = [int(b) for b in message_bits]
                if len(bits) < fixed_length:
                    bits = bits.tolist() + [0] * (fixed_length - len(bits))
                else:
                    bits = bits[:fixed_length]
                secrets.append(bits)
            secrets = np.array(secrets)
            # Convert secrets to the required format (e.g., extend to match model input if necessary)
            # Here, we assume the model expects a fixed size secret, adjust if necessary
            # For example, pad with zeros to a larger fixed size if the model requires
            # Here, we assume the 'secret' input can handle variable lengths or is appropriately sized

            # Prepare feed_dict
            feed_dict = {
                self.parent.input_secret: secrets,
                self.parent.input_image: images_np
            }

            # Run the session
            hidden_imgs, residuals = self.parent.sess.run(
                [self.parent.output_stegastamp, self.parent.output_residual],
                feed_dict=feed_dict
            )

            # hidden_imgs is numpy array of shape (batch_size, H, W, C)
            # resize H and W to match the input image size
            hidden_imgs = np.array([cv2.resize(hidden_img, (128, 128), interpolation=cv2.INTER_LINEAR) for hidden_img in hidden_imgs])
            # Convert back to torch tensor
            hidden_imgs_torch = torch.from_numpy(hidden_imgs).permute(0, 3, 1, 2).float()

            # Convert from [0, 1] to [-1, 1]
            hidden_imgs_torch = hidden_imgs_torch * 2 - 1
            # move to the same device as images
            hidden_imgs_torch = hidden_imgs_torch.to(images.device)


            return hidden_imgs_torch

    class Decoder:
        def __init__(self, parent):
            self.parent = parent

        def eval(self):
            pass  # Placeholder to match PyTorch model interface

        def __call__(self, images):
            """
            Decodes messages from images.

            Args:
                images (torch.Tensor): Batch of images with embedded messages, shape (batch_size, C, H, W).

            Returns:
                torch.Tensor: Batch of decoded binary messages with shape (batch_size, message_length).
            """
            # Convert to numpy array
            images_np = images.detach().cpu().numpy()

            # Convert from [-1, 1] to [0, 1]
            images_np = (images_np + 1) / 2.0

            # Resize images to (400, 400)
            batch_size = images_np.shape[0]
            resized_images = np.zeros((batch_size, 400, 400, 3))
            for i in range(batch_size):
                image = images_np[i].transpose(1, 2, 0)
                resized_image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_LINEAR)
                resized_images[i] = resized_image

            # Prepare feed_dict
            feed_dict = {
                self.parent.input_image: resized_images
            }

            # Run the session
            decoded_secrets = self.parent.sess.run(self.parent.output_secret, feed_dict=feed_dict)

            # decoded_secrets is numpy array of shape (batch_size, secret_length)
            # Adjust secret_length if necessary; assuming it matches message_length

            # For simplicity, assume the first 'message_length' bits are the message
            fixed_length = self.parent.message_length

            messages = []
            for secret_bits in decoded_secrets:
                bits = [int(round(bit)) for bit in secret_bits[:fixed_length]]
                # Ensure binary values
                bits = [1 if b > 0.5 else 0 for b in bits]
                messages.append(bits)

            # Convert messages to torch tensor
            messages_torch = torch.tensor(messages).float()
            # move to the same device as images
            messages_torch = messages_torch.to(images.device)

            return messages_torch


# Example usage:
if __name__ == "__main__":
    # Initialize the model with the path to your saved StegaStamp model
    model_path = '/scratch/qilong3/transferattack/targets/checkpoints/stegaStamp/stegastamp_pretrained'
    message_length = 30  # Define your desired message length
    model = StegaStampModel(model_path, message_length=message_length)

    # Example tensors (replace with actual data)
    batch_size = 4
    channels, height, width = 3, 256, 256
    images = torch.randn(batch_size, channels, height, width)  # Example image tensor
    messages = torch.randint(0, 2, (batch_size, message_length)).float()  # Example binary messages

    # Encode messages into images
    embedded_images = model.encoder(images, messages)
    print("Embedded Images Shape:", embedded_images.shape)

    # Decode messages from images
    decoded_messages = model.decoder(embedded_images)
    print("Decoded Messages Shape:", decoded_messages.shape)

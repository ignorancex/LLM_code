
import torch.nn as nn

from core.config import DEVICE
from core.optimization_vis import optim_gan, dot_cossim_loss, LossObject_Target, OptimGoal_ForwardHook

from core.utils import access_activations_forward_hook

from decoder.core import FeatureExtractorForwardWrapper



def generate_gan_visualizations(activation_vectors, decoder, feature_extractor):

    wrapped_model = FeatureExtractorForwardWrapper(feature_extractor)

    loss_function = dot_cossim_loss
    loss_object = LossObject_Target(loss_function, target=activation_vectors.float().to(DEVICE))
    optim_goal = OptimGoal_ForwardHook(wrapped_model.identity_layer_for_output_hook, loss_object, weight=-1.0)

    decoder_latent_hook = decoder.decoder.decoder.up_blocks[0]  # default is 2
    initial_gan_activations = access_activations_forward_hook([activation_vectors.float().to(DEVICE)], decoder.forward, decoder_latent_hook)
    visualizations = optim_gan(initial_gan_activations, decoder.forward_no_inputs, decoder_latent_hook,
                            wrapped_model, [optim_goal], from_single_vector=False, num_steps=32,
                            normalization_preprocessor=nn.Identity()).detach().to("cpu")


    wrapped_model = FeatureExtractorForwardWrapper(feature_extractor)
    loss_function = dot_cossim_loss
    loss_object = LossObject_Target(loss_function, target=activation_vectors.float().to(DEVICE))
    optim_goal = OptimGoal_ForwardHook(wrapped_model.identity_layer_for_output_hook, loss_object, weight=-1.0)

    decoder_latent_hook = decoder.decoder.decoder.up_blocks[3]  # default is 2
    initial_gan_activations = access_activations_forward_hook([activation_vectors.float().to(DEVICE)], decoder.forward, decoder_latent_hook)
    visualizations_2 = optim_gan(initial_gan_activations, decoder.forward_no_inputs, decoder_latent_hook,
                            wrapped_model, [optim_goal], from_single_vector=False, num_steps=128,
                            normalization_preprocessor=nn.Identity()).detach().to("cpu")


    return visualizations, visualizations_2


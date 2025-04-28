import xarray as xr
import sys
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from tensorflow.keras.callbacks import Callback
import numpy as np
import tensorflow as tf
from functools import partial

AUTOTUNE = tf.data.experimental.AUTOTUNE
from dask.diagnostics import ProgressBar
import pandas as pd
import tensorflow.keras.layers as layers
import json
from tensorflow.keras.optimizers import Adam
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras import layers

config_file = sys.argv[-1]  # configuratoin file for training algorithm
pretrained_unet = False  # set this as true if you want to use the same U-Net or a specific unet everytime.
with open(config_file, 'r') as f:
    config = json.load(f)
# Reviewer Comment
config["delta"] =1
factor = 3600 * 24
# set to 1 if using normal precip (kgs-1)
input_shape = config["input_shape"]  # the input shape of the reanalyses
output_shape = config["output_shape"]
# modified the output channels, filters, and output shape
n_filters = config["n_filters"]
kernel_size = config["kernel_size"]
n_channels = config["n_input_channels"]
n_output_channels = config["n_output_channels"]
BATCH_SIZE = 8#config["batch_size"]
init_weights = True
# config["itensity_weight"] = config["itensity_weight"]*2
if not init_weights:
    config["model_name"] = config["model_name"] + "_" + config["train_gcm"] + "_" + str(config["ad_loss_factor"])
# creating a path to store the model outputs
if not os.path.exists(f'{config["output_folder"]}/{config["model_name"]}'):
    os.makedirs(f'{config["output_folder"]}/{config["model_name"]}')
# custom modules
sys.path.append(os.getcwd())
from src.layers import *
from src.models import *
from src.gan import *
from src.process_input_training_data import *

stacked_X, y, vegt, orog, he = preprocess_input_data(config)
stacked_X = stacked_X.sel(GCM =config["train_gcm"])
y = y.sel(GCM =config["train_gcm"])[['pr']]

if config["time_period"] == "historical":
    y = y.sel(time = slice(None, "2014"))
    stacked_X = stacked_X.sel(time = y.time.to_index().intersection(stacked_X.time.to_index()))

# modified this line to retrain on a shorter period
elif config["time_period"] == "future only":
    y = y.sel(time = slice("2044", None))
    stacked_X = stacked_X.sel(time = y.time.to_index().intersection(stacked_X.time.to_index()))


with ProgressBar():
    y = y.load().transpose("time", "lat", "lon")
    stacked_X = stacked_X.transpose("time", "lat", "lon", "channel")
    stacked_X = stacked_X.load()

strategy = MirroredStrategy()

with strategy.scope():
    n_filters = n_filters#+ [512]
    generator = res_linear_activation_v2(input_shape, output_shape, n_filters[:],
                                      kernel_size, n_channels, n_output_channels,
                                      resize=True, final_activation=tf.keras.layers.LeakyReLU(0.6))

    unet_model = unet_linear_v2(input_shape, output_shape, n_filters,
                                 kernel_size, n_channels, n_output_channels,
                                 resize=True, final_activation=tf.keras.layers.LeakyReLU(0.2))

    noise_dim = [tuple(generator.inputs[i].shape[1:]) for i in range(len(generator.inputs) - 1)]
    d_model = get_discriminator_model_v1(tuple(output_shape) + (n_output_channels,),
                                      tuple(input_shape) + (n_channels,))
    learning_rate_adapted = False
    generator_checkpoint = GeneratorCheckpoint(
        generator=generator,
        filepath=f'{config["output_folder"]}/{config["model_name"]}/generator',
        period=5  # Save every 5 epochs
    )

    discriminator_checkpoint = DiscriminatorCheckpoint(
        discriminator=d_model,
        filepath=f'{config["output_folder"]}/{config["model_name"]}/discriminator',
        period=5  # Save every 5 epochs
    )

    unet_checkpoint = DiscriminatorCheckpoint(
        discriminator=unet_model,
        filepath=f'{config["output_folder"]}/{config["model_name"]}/unet',
        period=5  # Save every 5 epochs
    )


    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config["learning_rate_unet"], decay_steps=config["decay_steps"], decay_rate=config["decay_rate"])

    lr_schedule_gan = tf.keras.optimizers.schedules.ExponentialDecay(
        config["learning_rate"], decay_steps=config["decay_steps"], decay_rate=config["decay_rate_gan"])

    generator_optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule_gan, beta_1=config["beta_1"], beta_2=config["beta_2"])

    discriminator_optimizer = keras.optimizers.Adam(
        learning_rate=config["learning_rate"], beta_1=config["beta_1"], beta_2=config["beta_2"])
    unet_optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule, beta_1=config["beta_1"], beta_2=config["beta_2"])

    # Start training the model.
    # we normalize by a fixed normalization value
    total_size = stacked_X.time.size
    eval_times = BATCH_SIZE * (total_size//BATCH_SIZE)
    try:
        av_int_weight = config["av_int_weight"]

    except:
        av_int_weight =0.0
        config["av_int_weight"] = av_int_weight

    wgan = WGAN_Cascaded_IP(discriminator=d_model,
                                     generator=generator,
                                     latent_dim=noise_dim,
                                     discriminator_extra_steps=config["discrim_steps"],
                                     ad_loss_factor=config["ad_loss_factor"],
                                     orog=tf.convert_to_tensor(orog.values, 'float32'),
                                     vegt=tf.convert_to_tensor(vegt.values, 'float32'),
                                     he=tf.convert_to_tensor(he.values, 'float32'),
                                     gp_weight=config["gp_weight"],
                                     unet=unet_model,
                                     train_unet=True,
                                     intensity_weight=config["itensity_weight"],
                                     average_intensity_weight =av_int_weight)

    # Compile the WGAN model.
    wgan.compile(d_optimizer=discriminator_optimizer,
                 g_optimizer=generator_optimizer,
                 g_loss_fn=generator_loss,
                 d_loss_fn=discriminator_loss,
                 u_optimizer=unet_optimizer,
                 u_loss_fn=tf.keras.losses.mean_squared_error)

    # Start training the model.
    # we normalize by a fixed normalization value
    
    data = tuple([tf.convert_to_tensor(np.log(y.pr[:eval_times].transpose("time", "lat", "lon").values*factor + config["delta"]), 'float32'),
                  tf.convert_to_tensor(stacked_X[:eval_times].values, 'float32')])

    with open(f'{config["output_folder"]}/{config["model_name"]}/config_info.json', 'w') as f:
        json.dump(config, f)

    wgan.fit(data, batch_size=BATCH_SIZE, epochs=config["epochs"], verbose=1, shuffle=True,
             callbacks=[generator_checkpoint, discriminator_checkpoint, unet_checkpoint])

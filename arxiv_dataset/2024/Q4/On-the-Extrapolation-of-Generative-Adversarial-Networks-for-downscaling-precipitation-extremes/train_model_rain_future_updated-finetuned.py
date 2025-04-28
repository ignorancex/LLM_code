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

# set to 1 if using normal precip (kgs-1)
input_shape = config["input_shape"]  # the input shape of the reanalyses
output_shape = config["output_shape"]
# modified the output channels, filters, and output shape
n_filters = config["n_filters"]
kernel_size = config["kernel_size"]
n_channels = config["n_input_channels"]
n_output_channels = config["n_output_channels"]
BATCH_SIZE = 16#config["batch_size"]
init_weights = True
# config["itensity_weight"] = config["itensity_weight"]*2

#config["model_name"] = config["model_name"] + "_" + str(config["ad_loss_factor"]) + str(config["output_varname"])
# appending more gcms if the list is large
# for i in config["gcms_for_training_GAN"]:
#     config["model_name"] = config["model_name"] + '_' + i
# creating a path to store the model outputs
if not os.path.exists(f'{config["output_folder"]}/{config["model_name"]}'):
    os.makedirs(f'{config["output_folder"]}/{config["model_name"]}')
# custom modules
sys.path.append(os.getcwd())
from src.layers import *
from src.models import *
from src.gan import *
from src.process_input_training_data import *

gcms_for_training = config["gcms_for_training_GAN"]
stacked_X, y, vegt, orog, he = preprocess_input_data(config)
output_means = xr.open_dataset(config["means_output"])
output_stds = xr.open_dataset(config["stds_output"])
# To modify
#stacked_X = stacked_X.sel(GCM =config["gcms_for_training_GAN"][0])
#y = y.sel(GCM =config["gcms_for_training_GAN"][0])[['pr']]



try:
    output_varname = [config["output_varname"]]
except:
    output_varname = ["pr"]
    
if output_varname[0] =="pr":
    config["delta"] =1
    conversion_factor = 3600 * 24
    config['conversion_factor'] = conversion_factor 
    y[output_varname[0]] = np.log(y[output_varname[0]] * conversion_factor + 1)
    loss_weight =1
elif "sfcWind" in output_varname[0]:   
    y[output_varname[0]] = (y[output_varname[0]] - output_means['sfcWind'].mean()) / output_stds['sfcWind'].mean()
    loss_weight = 10
else:
    y[output_varname[0]] = (y[output_varname[0]] - output_means[output_varname[0]].mean()) / output_stds[output_varname[0]].mean()
    loss_weight = 10

common_times = stacked_X.time.to_index().intersection(y.time.to_index())
if config["period_start"] is not None:   
    stacked_X = stacked_X.sel(GCM=gcms_for_training, time=common_times).sel(time = slice(config["period_start"], None))
    y = y.sel(GCM=gcms_for_training, time=common_times).sel(time = slice(config["period_start"], None))
else:
    stacked_X = stacked_X.sel(GCM=gcms_for_training, time=common_times)
    y = y.sel(GCM=gcms_for_training, time=common_times)
    
y = y[output_varname[0]].transpose("time", "lat", "lon", "GCM")
stacked_X = stacked_X.transpose("time", "lat", "lon", "GCM", "channel")
# rounding to three decimal places
with ProgressBar():
    y = y.load()
    stacked_X = stacked_X.load()

strategy = MirroredStrategy()
if output_varname[0] =="pr":
    final_activation_unet = tf.keras.layers.LeakyReLU(0.01)
else:
    final_activation_unet = 'linear'
with strategy.scope():
    generator = tf.keras.models.load_model(f'{config["output_folder"]}/{config["model_name"]}/generator_final.h5',
                                           custom_objects={"BicubicUpSampling2D": BicubicUpSampling2D, "SymmetricPadding2D": SymmetricPadding2D})
    unet_model = tf.keras.models.load_model(f'{config["output_folder"]}/{config["model_name"]}/unet_final.h5',
                                            custom_objects={"BicubicUpSampling2D": BicubicUpSampling2D, "SymmetricPadding2D": SymmetricPadding2D})
    d_model = tf.keras.models.load_model(f'{config["output_folder"]}/{config["model_name"]}/discriminator_final.h5',
                                            custom_objects={"BicubicUpSampling2D": BicubicUpSampling2D, "SymmetricPadding2D": SymmetricPadding2D})

    noise_dim = [tuple(generator.inputs[i].shape[1:]) for i in range(len(generator.inputs) - 1)]
    d_model.summary()
    learning_rate_adapted = True
    generator_checkpoint = GeneratorCheckpoint(
        generator=generator,
        filepath=f'{config["output_folder"]}/{config["model_name"]}/generator_ft',
        period=5  # Save every 5 epochs
    )

    discriminator_checkpoint = DiscriminatorCheckpoint(
        discriminator=d_model,
        filepath=f'{config["output_folder"]}/{config["model_name"]}/discriminator_ft',
        period=5  # Save every 5 epochs
    )

    unet_checkpoint = DiscriminatorCheckpoint(
        discriminator=unet_model,
        filepath=f'{config["output_folder"]}/{config["model_name"]}/unet_ft',
        period=5  # Save every 5 epochs
    )

    config["learning_rate_unet"] = 2e-6
    config["learning_rate"] = 2e-6
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
    BATCH_SIZE = int(BATCH_SIZE/len(gcms_for_training))
    print("bs", BATCH_SIZE)
    eval_times =(BATCH_SIZE * ((total_size//2) // BATCH_SIZE))
    
    data = create_dataset(y, stacked_X, eval_times)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    data = data.with_options(options)

    data = data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    data = data.shuffle(16)
    
    

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
                                     average_intensity_weight =av_int_weight, varname = output_varname[0])
    prediction_callback = PredictionCallback(unet_model, generator, wgan, [stacked_X.isel(time = slice(0,30), GCM =0)],
                                             y.isel(time = slice(0,30), GCM =0), orog = orog.values, 
                                             save_dir = f'{config["output_folder"]}/{config["model_name"]}',
                                             output_mean =output_means[output_varname[0]].mean().values, output_std = output_stds[output_varname[0]].mean().values, varname = output_varname[0])
    # Compile the WGAN model.
    wgan.compile(d_optimizer=discriminator_optimizer,
                 g_optimizer=generator_optimizer,
                 g_loss_fn=generator_loss,
                 d_loss_fn=discriminator_loss,
                 u_optimizer=unet_optimizer,
                 u_loss_fn=tf.keras.losses.mean_squared_error)

    # Start training the model.
    # we normalize by a fixed normalization value
    
#     data = tuple([tf.convert_to_tensor(np.log(y.pr[:eval_times].transpose("time", "lat", "lon").values*factor + config["delta"]), 'float32'),
#                   tf.convert_to_tensor(stacked_X[:eval_times].values, 'float32')])

    with open(f'{config["output_folder"]}/{config["model_name"]}/config_info.json', 'w') as f:
        json.dump(config, f)

    wgan.fit(data, batch_size=BATCH_SIZE, epochs=config["epochs"], verbose=1, shuffle=True,
             callbacks=[generator_checkpoint, discriminator_checkpoint, unet_checkpoint, prediction_callback])

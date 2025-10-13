import os
cdir = os.getcwd()
os.chdir(cdir)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
from tensorflow import keras
from keras import layers
import wandb
import numpy as np
import pandas as pd
from LDF_PINTO_Pde import PdeModel
from utils import get_ibc_and_inner_data

np.random.seed(1234)
tf.random.set_seed(1234)

# Data PreProcessing
# getting the data in a format to train the model from util.py file

# hyperparameters for data generation
boundary_samples = 100
tv = 1  # top velocity
train_vel = [1, 2, 3]
test_vel = [1, 1.5, 2, 2.5, 3, 3.5, 4]
seq_len = 40
domain_samples = 5000

data_dir = {"1": '../../CFD_data/u_1.mat',
            "1.5": '../../CFD_data/u_1.5.mat',
            "2": '../../CFD_data/u_2.mat',
            "2.5": '../../CFD_data/u_2.5.mat',
            "3": '../../CFD_data/u_3.mat',
            "3.5": '../../CFD_data/u_3.5.mat',
            "4": '../../CFD_data/u_4.mat',
            }
param_dir = '../../CFD_data/param.mat'

(xd, yd, xb, yb, u, v, x_top, y_top, u_top, v_top,
 xbc_in, ybc_in, ubc_in, vbc_in, xbc_b, ybc_b, ubc_b, vbc_b,
 xbc_top, ybc_top, ubc_top, vbc_top, sen_data) = get_ibc_and_inner_data(start=0., stop=1.,
                                                                        boundary_samples=boundary_samples,
                                                                        domain_samples=domain_samples,
                                                                        seq_len=seq_len,
                                                                        top_velocities=train_vel)

ivals = {'xin': xd, 'yin': yd, 'xb': xb, 'yb': yb, 'x_top': x_top, 'y_top': y_top,
         'xbc_in': xbc_in, 'ybc_in': ybc_in, 'ubc_in': ubc_in, 'vbc_in': vbc_in,
         'xbc_b': xbc_b, 'ybc_b': ybc_b, 'ubc_b': ubc_b, 'vbc_b': vbc_b,
         'xbc_top': xbc_top, 'ybc_top': ybc_top, 'ubc_top': ubc_top, 'vbc_top': vbc_top}
ovals = {'ub': u, 'vb': v, 'u_top': u_top, 'v_top': v_top}
parameters = {'nue': 0.02}

# Building the PINTO model using Functional API
initializer = tf.keras.initializers.GlorotNormal(1234)


# Define the sequential model for query value transfer
def get_model(model_name, layer_names, layer_units, activation='swish'):
    sq = keras.Sequential(name=model_name)
    for i in range(len(layer_units)):
        sq.add(layers.Dense(units=layer_units[i], kernel_initializer=initializer,
                            name=layer_names + str(i + 1)))
        sq.add(layers.Activation(activation=activation))
    return sq


# Lifting operation for query points
input1 = keras.Input(shape=(1,), name='X_layer')
rescale_input1 = layers.Rescaling(scale=2, offset=-1)(input1)
input2 = keras.Input(shape=(1,), name='Y_layer')
rescale_input2 = layers.Rescaling(scale=2, offset=-1)(input2)
sp_trans = get_model(model_name='spatial_transformation',
                     layer_names='spatial_layer',
                     layer_units=[64, 64], activation='swish')

sp = layers.Concatenate()([rescale_input1, rescale_input2])
sp = layers.Reshape(target_shape=(1, -1))(sp)
spq = sp_trans(sp)
residual = spq

# MLP for BPE units
input3 = layers.Input(shape=(None, 1,), name='Xbc_layer')
rescale_input3 = layers.Rescaling(scale=2, offset=-1)(input3)
input4 = layers.Input(shape=(None, 1,), name='ybc_layer')
rescale_input4 = layers.Rescaling(scale=2, offset=-1)(input4)

pe = layers.Concatenate()([rescale_input3, rescale_input4])
pe = get_model(model_name='BPE',
               layer_names='bpe_layer',
               layer_units=[64, 64], activation='swish')(pe)

# MLP for BVE units
input5 = layers.Input(shape=(None, 1,), name='ubc_layer')
rescale_input5 = layers.Rescaling(scale=1/3, offset=0)(input5)

ce = get_model(model_name='BVE',
               layer_names='bve_layer',
               layer_units=[64, 64], activation='swish')(rescale_input5)

# cross-attention layers
ct = layers.MultiHeadAttention(num_heads=2, key_dim=64)(query=spq, key=pe, value=ce)
ct = layers.Add()([residual, ct])
residual = ct
ct = layers.Dense(units=64, kernel_initializer=initializer, activation='swish')(ct)
ct = layers.Dense(units=64, kernel_initializer=initializer, activation='swish')(ct)
ct = layers.Add()([residual, ct])
ct = layers.Flatten()(ct)
residual = ct

# Projection operations for output functional spaces
# output model for u-function
ou = get_model(model_name='U', layer_units=[128, 128],
               layer_names='ou', activation='swish')(ct)
ou = layers.Dense(units=1, kernel_initializer=initializer, name='output_u')(ou)

# output model for v-function
ov = get_model(model_name='V', layer_units=[128, 128],
               layer_names='ov', activation='swish')(ct)
ov = layers.Dense(units=1, kernel_initializer=initializer, name='output_v')(ov)

# output model for p-function
op = get_model(model_name='P', layer_units=[64, 64],
               layer_names='op', activation='swish')(ct)
op = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, name='output_p')(op)

# building the PINTO model
model = keras.Model([input1, input2, input3, input4, input5],
                    [ou, ov, op])

model.summary()

# Training the model
initial_learning_rate = 1e-3

## uncomment the learning rate scheduler you want to use
# # Piecewise constant decay learning rate scheduler
boundaries = [5000, 10000]
values = [1e-3, 1e-4, 1e-5]
lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
     boundaries=boundaries, values=values)

# decay_steps = 10000
# decay_rate = 0.5
# staircase = True
# step_lim = 50000

## Modified exponential decay learning rate scheduler
# lr_schedule = SedLrSchedule(
#     initial_learning_rate=initial_learning_rate,
#     decay_steps=decay_steps,
#     higher_decay_rate=decay_rate[0],
#     lower_decay_rate = decay_rate[1],
#     staircase=staircase, step_lim=step_lim)

## Exponential decay learning rate scheduler
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=initial_learning_rate,
#     decay_steps=decay_steps,
#    decay_rate=decay_rate,
#     staircase=staircase)

# initiating the loss function and optimizer
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss=loss_fn)
model_dict = {"nn_model": model}
# metrics to track the traning process
metrics = {"loss": keras.metrics.Mean(name='loss'),
           "boundary_loss": keras.metrics.Mean(name='std_loss'),
           "residual_loss": keras.metrics.Mean(name='residual_loss'),
           "u_loss": keras.metrics.Mean(name='u_loss'),
           "v_loss": keras.metrics.Mean(name='v_loss')
           }
batches = 5

# initiating the PdeModel class
cm = PdeModel(inputs=ivals, outputs=ovals, get_models=model_dict, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics,
              parameters=parameters, train_vel=train_vel, test_vel=test_vel, batches=batches)

epochs = 50000
vf = 100  # verbose frequency
pf = 1000  # plot frequency
wb = False  # wandb logging

configuration = {'#_total_initial_and_boundary_points': len(xb),
                 '#_total_domain_points': len(xd),
                 "optimizer": 'Adam',
                 "batches": batches,
                 "learning_rate": initial_learning_rate,
                 "lr_schedule": 'Piecewise constant Decay scheduler',
                 # "decay_steps": decay_steps,
                 # "decay_rate": decay_rate,
                 # "staircase": staircase,
                 "Epochs": epochs,
                 "Activation": 'swish',
                 "model_name": 'LDF_PINTO_model',
                 "trainable_parameters": np.sum([np.prod(lay.shape) for lay in model.trainable_weights]),
                 "non_trainable_parameters": np.sum([np.prod(lay.shape) for lay in model.non_trainable_weights]),
                 "kernel_initializer": 'GlorotNormal',
                 "sequence_length": seq_len,
                 "nue": parameters['nue'],
                 "train_vel": train_vel,
                 "test_vel": test_vel}

print(configuration)
if wb:
    wandb.init(project='Finalising_results', config=configuration)

log_dir = 'output/LDF_PINTO/'
try:
    os.makedirs(log_dir)
except FileExistsError:
    pass
history = cm.run(epochs=epochs, log_dir=log_dir, data_dir=data_dir, param_dir=param_dir,
                 wb=wb, verbose_freq=vf, plot_freq=pf)
sdata = pd.DataFrame(sen_data)
sdata.to_csv(path_or_buf=log_dir + 'sensor.csv')
if wb:
    wandb.finish()

# Evaluation
cm.nn_model.save(log_dir + 'LDF_PINTO_model.keras')

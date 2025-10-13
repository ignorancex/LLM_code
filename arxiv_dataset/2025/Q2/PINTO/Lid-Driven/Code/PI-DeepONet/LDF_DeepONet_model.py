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
from LDF_DeepONet_Pde import PdeModel
from utils import get_ibc_and_inner_data

np.random.seed(1234)
tf.random.set_seed(1234)

# Data PreProcessing
boundary_samples = 100
tv = 1
train_vel = [1, 2, 3]
test_vel = [1.5, 2.5, 3.5, 4]
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

# Building the DeepONets model using Functional API
initializer = tf.keras.initializers.GlorotNormal(1234)


# Define the sequential model for query value transfer
def get_model(model_name, layer_names, layer_units, activation='swish'):
    sq = keras.Sequential(name=model_name)
    for i in range(len(layer_units)):
        sq.add(layers.Dense(units=layer_units[i], kernel_initializer=initializer,
                            name=layer_names + str(i + 1)))
        sq.add(layers.Activation(activation=activation))
    return sq


# Branch network for the input data
input1 = keras.Input(shape=(1,), name='X_layer')
rescale_input1 = layers.Rescaling(scale=2 / 1.5, offset=-0.5 / 1.5)(input1)
input2 = keras.Input(shape=(1,), name='Y_layer')
rescale_input2 = layers.Rescaling(scale=1, offset=-0.5)(input2)
sp_trans = get_model(model_name='spatial_transformation',
                     layer_names='spatial_layer',
                     layer_units=[64, 64, 64, 64, 64, 64], activation='swish')

sp = layers.Concatenate()([rescale_input1, rescale_input2])
spq = sp_trans(sp)
residual = spq

# Trunck network
input4 = keras.Input(shape=(seq_len,), name='Ubc_layer')
rescale_input4 = layers.Rescaling(scale=1/3, offset=0)(input4)

# output model for u-function
ou = get_model(model_name='U',
               layer_units=[64, 64, 64, 64, 64, 64], layer_names='ou', activation='swish')(rescale_input4)
ou = layers.Dot(axes=(1, 1))([ou, spq])

# output model for v-function
ov = get_model(model_name='V',
               layer_units=[64, 64, 64, 64, 64, 64], layer_names='ov', activation='swish')(rescale_input4)
ov = layers.Dot(axes=(1, 1))([ov, spq])

# output model for p-function
op = get_model(model_name='P',
               layer_units=[64, 64, 64, 64, 64, 64], layer_names='op', activation='swish')(rescale_input4)
op = layers.Dot(axes=(1, 1))([op, spq])

# Building the DeepONets model
model = keras.Model([input1, input2, input4], [ou, ov, op])

model.summary()

# Training the model
initial_learning_rate = 1e-3

boundaries = [5000, 10000]
values = [1e-3, 1e-4, 1e-5]

lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
     boundaries=boundaries, values=values)

# decay_steps = 10000
# decay_rate = 0.5
# staircase = True
# step_lim = 50000

# lr_schedule = SedLrSchedule(
#     initial_learning_rate=initial_learning_rate,
#     decay_steps=decay_steps,
#     higher_decay_rate=decay_rate[0],
#     lower_decay_rate = decay_rate[1],
#     staircase=staircase, step_lim=step_lim)

# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=initial_learning_rate,
#     decay_steps=decay_steps,
#    decay_rate=decay_rate,
#     staircase=staircase)

# Training the model
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss=loss_fn)
model_dict = {"nn_model": model}
metrics = {"loss": keras.metrics.Mean(name='loss'),
           "boundary_loss": keras.metrics.Mean(name='std_loss'),
           "residual_loss": keras.metrics.Mean(name='residual_loss'),
           "u_loss": keras.metrics.Mean(name='u_loss'),
           "v_loss": keras.metrics.Mean(name='v_loss')
           }
batches = 5

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
                 "model_name": 'LDF_Deeponet_model.keras',
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

log_dir = 'output/LDF_DeepONet/'
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
cm.nn_model.save(log_dir + 'LDF_Deeponet_model.keras')

import os
get_wd = os.getcwd()
os.chdir(get_wd)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import wandb
import numpy as np
import pandas as pd
from utils import get_train_data
from be1d_DeepONet_Pde import PdeModel

np.random.seed(1234)
tf.random.set_seed(1234)

# Data PreProcessing
# getting data in required format from utils.py
data_dir = None  # change the directory to your local directory where PDEBENCH Advection file presents.

# hyperparameters for data generation
train_indices = np.arange(80)
test_indices = [10, 20, 85, 99]
val_indices = np.arange(80, 100)
domain_samples = 1000
seq_len = 80

(xd, td, xb, tb, ub, x_init, t_init, u_init,
 xbc_in, tbc_in, ubc_in, xbc_b, tbc_b, ubc_b, xbc_init, tbc_init, ubc_init,
 idx_si, xval, tval, uval, xbc_val, tbc_val, ubc_val) = get_train_data(
    data_dir, seq_len=seq_len, domain_samples=domain_samples, indices=train_indices,
    val_indices=val_indices)

ivals = {'xin': xd, 'tin': td, 'xb': xb, 'tb': tb, 'xbc_in': xbc_in, 'tbc_in': tbc_in, 'ubc_in': ubc_in,
         'xbc_b': xbc_b, 'tbc_b': tbc_b, 'ubc_b': ubc_b, 'x_init': x_init, 't_init': t_init,
         'xbc_init': xbc_init, 'tbc_init': tbc_init, 'ubc_init': ubc_init,
         'xval': xval, 'tval': tval, 'xbc_val': xbc_val, 'tbc_val': tbc_val, 'ubc_val': ubc_val}
ovals = {'ub': ub, 'u_init': u_init, 'uval': uval}
parameters = {'nue': 0.01, 'test_ind': test_indices}

# Building the DeepONets model using functional API
initializer = tf.keras.initializers.GlorotUniform(seed=1234)


def get_model(model_name, layer_names, layer_units, activation='swish'):
    sq = keras.Sequential(name=model_name)
    for i in range(len(layer_units)):
        sq.add(layers.Dense(units=layer_units[i], kernel_initializer=initializer,
                            name=layer_names + str(i + 1)))
        sq.add(layers.Activation(activation=activation))
    return sq


# Branch network
input1 = layers.Input(shape=(1,), name='x_input')
rescale_input1 = layers.Rescaling(scale=2.0, offset=-1.)(input1)
input2 = layers.Input(shape=(1,), name='t_input')
rescale_input2 = layers.Rescaling(scale=2.0, offset=-1.)(input2)
sp_trans = get_model(model_name='spatial_transformation',
                     layer_names='spatial_layer',
                     layer_units=[128, 128, 128, 128, 128, 128, 128], activation='tanh')

sp = layers.Concatenate()([rescale_input1, rescale_input2])
spq = sp_trans(sp)

# Trunck Network
input3 = keras.Input(shape=(seq_len,), name='ubc_layer')

ou = get_model(model_name='U',
               layer_units=[128, 128, 128, 128, 128, 128, 128], layer_names='ou', activation='tanh')(input3)
# output function
ou = layers.Dot(axes=(1, 1))([ou, spq])
# buidling DeepONets model
model = keras.Model([input1, input2, input3], [ou])

model.summary()

# Training the model
initial_learning_rate = 1e-3

decay_steps = 10000
decay_rate = 0.9
staircase = True
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=staircase)

loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)  # change learning rate argument accordingly
model_dict = {"nn_model": model}
metrics = {"loss": keras.metrics.Mean(name='loss'),
           "init_loss": keras.metrics.Mean(name='init_loss'),
           "bound_loss": keras.metrics.Mean(name='bound_loss'),
           "residual_loss": keras.metrics.Mean(name='residual_loss'),
           "val_loss": keras.metrics.Mean(name='val_loss'),
           "val_data_loss": keras.metrics.Mean(name='val_data_loss'),
           "val_res_loss": keras.metrics.Mean(name='val_res_loss')
           }
batches = 6
cm = PdeModel(inputs=ivals, outputs=ovals, get_models=model_dict, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics,
              parameters=parameters, batches=batches)

epochs = 20000
vf = 100
pf = 1000
wb = True

configuration = {
    '#_total_initial_and_boundary_points': len(xb),
    '#_total_domain_points': len(xd),
    "optimizer": "Adam",
    'initial_learning_rate': initial_learning_rate,
    'lr_Schedule': 'Exponential Decay',
    'decay_steps': decay_steps,
    'decay_rate': decay_rate,
    'staircase': staircase,
    "batches": batches,
    "Epochs": epochs,
    "Activation": 'tanh',
    "model_name": 'Burgers_DeepONet_model.keras',
    "trainable_parameters": np.sum([np.prod(lay.shape) for lay in model.trainable_weights]),
    "non_trainable_parameters": np.sum([np.prod(lay.shape) for lay in model.non_trainable_weights]),
    "training_strategy": 'normal training',
    'train_indices': train_indices,
    'test_indices': test_indices}

print(configuration)

if wb:
    wandb.init(project='Finalising_results', config=configuration)

log_dir = 'output/Burgers_DeepONet/'
try:
    os.makedirs(log_dir)
except FileExistsError:
    pass

history = cm.run(epochs=epochs, idx_si=idx_si, ddir=data_dir, log_dir=log_dir,
                 wb=wb, verbose_freq=vf, plot_freq=pf)
sdata = pd.DataFrame({'sensor_indices': idx_si.flatten()})
sdata.to_csv(path_or_buf=log_dir + 'sensor.csv')

if wb:
    wandb.finish()

# Evaluation
cm.nn_model.save(log_dir + 'Burgers_Deeponet_model.keras')

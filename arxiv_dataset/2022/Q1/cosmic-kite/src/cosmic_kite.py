import os
import pkg_resources
import numpy as np

location  = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(location, 'data')

# Let's load the scaler
from pickle import load
from sklearn import preprocessing

scaler_x_TT = load(open(data_path + '/scaler_x_TT.pkl', 'rb'))
scaler_y_TT = load(open(data_path + '/scaler_y_TT.pkl', 'rb'))
scaler_x_EE = load(open(data_path + '/scaler_x_EE.pkl', 'rb'))
scaler_y_EE = load(open(data_path + '/scaler_y_EE.pkl', 'rb'))
scaler_x_TE = load(open(data_path + '/scaler_x_TE.pkl', 'rb'))
scaler_y_TE = load(open(data_path + '/scaler_y_TE.pkl', 'rb'))

# Let's load some neccesary libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU

# Let's define the architecture and load the trained values

# Armo los tensores input
lmin              = 2
lmax              = 2650
actFun            = 'LeakyReLU'
batch_size        = 256
original_dim      = 1*(lmax-lmin)
latent_dim        = 6
intermediate_dim  = 1000
intermediate_dim2 = 500
intermediate_dim3 = 100
intermediate_dim4 = 50
epochs            = 100
epsilon_std       = 1.0
input_shape       = (original_dim,)

# TT VAE model = encoder + decoder
#{{{
# build encoder model
inputs    = Input(shape=input_shape, name='encoder_input')
x         = Dense(intermediate_dim)(inputs)
x         = LeakyReLU(alpha=0.1)(x)
x         = Dense(intermediate_dim3)(x)
x         = LeakyReLU(alpha=0.1)(x)
x         = Dense(intermediate_dim4)(x)
x         = LeakyReLU(alpha=0.1)(x)
latent    = Dense(latent_dim, name='z_mean', activation='linear')(x)

# instantiate encoder model
encoder_TT = Model(inputs, latent, name='encoder')

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x             = Dense(intermediate_dim4)(latent_inputs)
x             = LeakyReLU(alpha=0.1)(x)
x             = Dense(intermediate_dim3)(x)
x             = LeakyReLU(alpha=0.1)(x)
x             = Dense(intermediate_dim)(x)
x             = LeakyReLU(alpha=0.1)(x)
outputs       = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder_TT = Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
dec_output = decoder_TT(latent)
enc_output = encoder_TT(inputs)
vae_TT        = Model(inputs, [enc_output, dec_output], name='vae_mlp')

vae_TT.load_weights(data_path + '/vae_model_TT.h5')
#}}}
# EE VAE model = encoder + decoder
#{{{
# build encoder model
inputs    = Input(shape=input_shape, name='encoder_input')
x         = Dense(intermediate_dim)(inputs)
x         = LeakyReLU(alpha=0.1)(x)
x         = Dense(intermediate_dim3)(x)
x         = LeakyReLU(alpha=0.1)(x)
x         = Dense(intermediate_dim4)(x)
x         = LeakyReLU(alpha=0.1)(x)
latent    = Dense(latent_dim, name='z_mean', activation='linear')(x)

# instantiate encoder model
encoder_EE = Model(inputs, latent, name='encoder')

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x             = Dense(intermediate_dim4)(latent_inputs)
x             = LeakyReLU(alpha=0.1)(x)
x             = Dense(intermediate_dim3)(x)
x             = LeakyReLU(alpha=0.1)(x)
x             = Dense(intermediate_dim)(x)
x             = LeakyReLU(alpha=0.1)(x)
outputs       = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder_EE = Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
dec_output = decoder_EE(latent)
enc_output = encoder_EE(inputs)
vae_EE        = Model(inputs, [enc_output, dec_output], name='vae_mlp')

vae_EE.load_weights(data_path + '/vae_model_EE.h5')
#}}}
# TE VAE model = encoder + decoder
#{{{
# build encoder model
inputs    = Input(shape=input_shape, name='encoder_input')
x         = Dense(intermediate_dim)(inputs)
x         = LeakyReLU(alpha=0.1)(x)
x         = Dense(intermediate_dim3)(x)
x         = LeakyReLU(alpha=0.1)(x)
x         = Dense(intermediate_dim4)(x)
x         = LeakyReLU(alpha=0.1)(x)
latent    = Dense(latent_dim, name='z_mean', activation='linear')(x)

# instantiate encoder model
encoder_TE = Model(inputs, latent, name='encoder')

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x             = Dense(intermediate_dim4)(latent_inputs)
x             = LeakyReLU(alpha=0.1)(x)
x             = Dense(intermediate_dim3)(x)
x             = LeakyReLU(alpha=0.1)(x)
x             = Dense(intermediate_dim)(x)
x             = LeakyReLU(alpha=0.1)(x)
outputs       = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder_TE = Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
dec_output = decoder_TE(latent)
enc_output = encoder_TE(inputs)
vae_TE        = Model(inputs, [enc_output, dec_output], name='vae_mlp')

vae_TE.load_weights(data_path + '/vae_model_TE.h5')
#}}}

# Let's define de main functions

def pars2ps(pars):
  #if ((np.max(pars[:,0]) > 0.023482) or (np.min(pars[:,0]) < 0.021282)): print('WARNING! Some values of omega_b*h2 are outside the training range')
  #if ((np.max(pars[:,1]) > 0.130607) or (np.min(pars[:,1]) < 0.109607)): print('WARNING! Some values of omega_c*h2 are outside the training range')
  #if ((np.max(pars[:,2]) > 76.52) or (np.min(pars[:,2]) < 58.121)): print('WARNING! Some values of H0 are outside the training range')
  #if ((np.max(pars[:,3]) > 0.99454) or (np.min(pars[:,3]) < 0.937550)): print('WARNING! Some values of n are outside the training range')
  #if ((np.max(pars[:,4]) > 0.094) or (np.min(pars[:,4]) < 0.01430)): print('WARNING! Some values of tau are outside the training range')
  #if ((np.max(pars[:,5]) > 2.2705e-9) or (np.min(pars[:,5]) < 1.9305e-9)): print('WARNING! Some values of As are outside the training range')
  pars_scaled    = scaler_y_TT.transform(pars)
  ell_array      = np.arange(lmin, lmax)
  tt_pred_scaled = decoder_TT.predict(pars_scaled)
  tt_pred        = scaler_x_TT.inverse_transform(tt_pred_scaled)[:,0:(lmax-lmin)]
  ee_pred_scaled = decoder_EE.predict(pars_scaled)
  ee_pred        = scaler_x_EE.inverse_transform(ee_pred_scaled)[:,0:(lmax-lmin)]
  te_pred_scaled = decoder_TE.predict(pars_scaled)
  te_pred        = scaler_x_TE.inverse_transform(te_pred_scaled)[:,0:(lmax-lmin)]
  return [tt_pred, ee_pred, te_pred, ell_array]


def tt2pars(ps):
  ps_scaled        = scaler_x_TT.transform(ps)
  pars_pred_scaled = encoder_TT.predict(ps_scaled)
  pars_pred        = scaler_y_TT.inverse_transform(pars_pred_scaled)
  return pars_pred  

def ee2pars(ps):
  ps_scaled        = scaler_x_EE.transform(ps)
  pars_pred_scaled = encoder_EE.predict(ps_scaled)
  pars_pred        = scaler_y_EE.inverse_transform(pars_pred_scaled)
  return pars_pred  

def te2pars(ps):
  ps_scaled        = scaler_x_TE.transform(ps)
  pars_pred_scaled = encoder_TE.predict(ps_scaled)
  pars_pred        = scaler_y_TE.inverse_transform(pars_pred_scaled)
  return pars_pred  


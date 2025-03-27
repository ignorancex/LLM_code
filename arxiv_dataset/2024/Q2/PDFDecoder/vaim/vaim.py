import numpy as np
import tensorflow as tf

from tensorflow.keras import layers,optimizers,losses,models
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

class Sampling(layers.Layer):

    def __init__(self):
        super(Sampling, self).__init__()

    def call(self, input_data):
        z_mean, z_log_var = input_data
        input_batch_size = tf.shape(z_mean)[0]
        input_feature_dim = tf.shape(z_mean)[1]
        eps = tf.random.uniform(shape=(input_batch_size, input_feature_dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * eps
        return z


class VAIM(Model):

    def __init__(self, latent_dim):
        super(VAIM, self).__init__()

        # Size of what's going into the autoencoder
        self.input_dim = 28*28

        #Size of the observable dimension
        self.observable_dim = (32,)

        # Size of the latent dimension
        self.latent_dim = latent_dim

        # Build encoder and decoder networks and model
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        # Track training losses in network
        self.total_loss_tracker = tf.keras.metrics.Mean(name='train_total_loss')
        self.reco_loss_tracker = tf.keras.metrics.Mean(name='train_reco_loss')
        self.latent_loss_tracker = tf.keras.metrics.Mean(name='train_latent_loss')
        self.kl_loss_tracker = tf.keras.metrics.Mean(name='train_kl_loss')

        # Track validation losses in network
        self.val_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
        self.val_reco_loss_tracker = tf.keras.metrics.Mean(name='reco_loss')
        self.val_latent_loss_tracker = tf.keras.metrics.Mean(name='latent_loss')
        self.val_kl_loss_tracker = tf.keras.metrics.Mean(name='kl_loss')

        self.beta = 1.

    def build_encoder(self):
        encoder_inputs = layers.Input(shape=(self.input_dim,))

        x1 = layers.Dense(units=2048, kernel_regularizer=l2(1e-6))(encoder_inputs)
        x2 = layers.Activation('ELU')(x1)
        x2 = layers.Dense(units=2048, kernel_regularizer=l2(1e-6))(x2)
        x2 = layers.Activation('ELU')(x2)
        ##
        x3 = layers.Dense(units=1024, kernel_regularizer=l2(1e-6))(x2)
        sc1 = layers.Dense(units=1024, kernel_regularizer=l2(1e-6))(x1)
        sc1 = layers.Add()([sc1, x3])
        x3 = layers.Activation('ELU')(sc1)
        x4 = layers.Dense(units=1024, kernel_regularizer=l2(1e-6))(x3)
        sc2 = layers.Dense(units=1024, kernel_regularizer=l2(1e-6))(x2)
        sc2 = layers.Add()([sc2, x4])
        x4 = layers.Activation('ELU')(sc2)
        ##
        x5 = layers.Dense(units=512, kernel_regularizer=l2(1e-6))(x4)
        sc3 = layers.Dense(units=512, kernel_regularizer=l2(1e-6))(x3)
        sc3 = layers.Add()([sc3, x5])
        x5 = layers.Activation('ELU')(sc3)
        x6 = layers.Dense(units=512, kernel_regularizer=l2(1e-6))(x5)
        sc4 = layers.Dense(units=512, kernel_regularizer=l2(1e-6))(x4)
        sc4 = layers.Add()([sc4, x6])
        x6 = layers.Activation('ELU')(sc4)
        ##
        x7 = layers.Dense(units=256, kernel_regularizer=l2(1e-6))(x6)
        sc5 = layers.Dense(units=256, kernel_regularizer=l2(1e-6))(x5)
        sc5 = layers.Add()([sc5, x7])
        x7 = layers.Activation('ELU')(sc5)
        x8 = layers.Dense(units=256, kernel_regularizer=l2(1e-6))(x7)
        sc6 = layers.Dense(units=256, kernel_regularizer=l2(1e-6))(x6)
        sc6 = layers.Add()([sc6, x8])
        x8 = layers.Activation('ELU')(sc6)
        ##
        x9 = layers.Dense(units=128, kernel_regularizer=l2(1e-6))(x8)
        sc7 = layers.Dense(units=128, kernel_regularizer=l2(1e-6))(x7)
        sc7 = layers.Add()([sc7, x9])
        x9 = layers.Activation('ELU')(sc7)
        x10 = layers.Dense(units=128, kernel_regularizer=l2(1e-6))(x9)
        x10 = layers.Activation('ELU')(x10)

        self.z_mean = layers.Dense(units=self.latent_dim,
                                   activation='linear',
                                   name='z_mean')(x10)
        self.z_log_var = layers.Dense(units=self.latent_dim,
                                      activation='linear',
                                      name='z_log_var')(x10)
        self.z_obs = layers.Dense(units=self.observable_dim[0],
                                  activation='linear',
                                  name='z_obs')(x10)

        self.z_sampling = Sampling()([self.z_mean, self.z_log_var])

        encoder_model = models.Model(inputs=encoder_inputs,
                                     outputs=[self.z_mean,
                                              self.z_log_var,
                                              self.z_sampling,
                                              self.z_obs],
                                     name='encoder')

        return encoder_model

    def build_decoder(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        observable_inputs = layers.Input(shape=self.observable_dim)
        decoder_inputs = layers.concatenate([latent_inputs,observable_inputs],
                                             axis=1)

        x1 = layers.Dense(units=128, kernel_regularizer=l2(1e-6))(decoder_inputs)
        x2 = layers.Activation('ELU')(x1)
        x2 = layers.Dense(units=128, kernel_regularizer=l2(1e-6))(x2)
        x2 = layers.Activation('ELU')(x2)
        ##
        x3 = layers.Dense(units=256, kernel_regularizer=l2(1e-6))(x2)
        sc1 = layers.Dense(units=256, kernel_regularizer=l2(1e-6))(x1)
        sc1 = layers.Add()([sc1, x3])
        x3 = layers.Activation('ELU')(sc1)
        x4 = layers.Dense(units=256, kernel_regularizer=l2(1e-6))(x3)
        sc2 = layers.Dense(units=256, kernel_regularizer=l2(1e-6))(x2)
        sc2 = layers.Add()([sc2, x4])
        x4 = layers.Activation('ELU')(sc2)
        ##
        x5 = layers.Dense(units=512, kernel_regularizer=l2(1e-6))(x4)
        sc3 = layers.Dense(units=512, kernel_regularizer=l2(1e-6))(x3)
        sc3 = layers.Add()([sc3, x5])
        x5 = layers.Activation('ELU')(sc3)
        x6 = layers.Dense(units=512, kernel_regularizer=l2(1e-6))(x5)
        sc4 = layers.Dense(units=512, kernel_regularizer=l2(1e-6))(x4)
        sc4 = layers.Add()([sc4, x6])
        x6 = layers.Activation('ELU')(sc4)
        ##
        x7 = layers.Dense(units=1024, kernel_regularizer=l2(1e-6))(x6)
        sc5 = layers.Dense(units=1024, kernel_regularizer=l2(1e-6))(x5)
        sc5 = layers.Add()([sc5, x7])
        x7 = layers.Activation('ELU')(sc5)
        x8 = layers.Dense(units=1024, kernel_regularizer=l2(1e-6))(x7)
        sc6 = layers.Dense(units=1024, kernel_regularizer=l2(1e-6))(x6)
        sc6 = layers.Add()([sc6, x8])
        x8 = layers.Activation('ELU')(sc6)
        ##
        x9 = layers.Dense(units=2048, kernel_regularizer=l2(1e-6))(x8)
        sc7 = layers.Dense(units=2048, kernel_regularizer=l2(1e-6))(x7)
        sc7 = layers.Add()([sc7, x9])
        x9 = layers.Activation('ELU')(sc7)
        x10 = layers.Dense(units=2048, kernel_regularizer=l2(1e-6))(x9)
        x10 = layers.Activation('ELU')(x10)

        decoder_outputs = layers.Dense(units=self.input_dim,
                                       activation='linear')(x10)

        decoder_model = models.Model(inputs=[latent_inputs,observable_inputs],
                                     outputs=decoder_outputs,
                                     name='decoder')

        return decoder_model

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reco_loss_tracker,
                self.latent_loss_tracker,
                self.kl_loss_tracker,
                self.val_loss_tracker,
                self.val_reco_loss_tracker,
                self.val_latent_loss_tracker,
                self.val_kl_loss_tracker]

    @tf.function
    def train_step(self, inputs):
        x_data, y_data = inputs
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z_sampling, z_obs = self.encoder(x_data)
            reco = self.decoder([z_sampling, z_obs])
            reco_loss = tf.reduce_mean(losses.mean_squared_error(x_data, reco))
            latent_loss = tf.reduce_mean(losses.mean_squared_error(z_obs, y_data))
            kl_loss = tf.reduce_mean(-0.5 * (1. + z_log_var - tf.square(z_mean)
                                             - tf.exp(z_log_var)))
            total_loss = reco_loss + self.beta*kl_loss + latent_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reco_loss_tracker.update_state(reco_loss)
        self.latent_loss_tracker.update_state(latent_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            'train_total_loss': self.total_loss_tracker.result(),
            'train_reco_loss': self.reco_loss_tracker.result(),
            'train_latent_loss': self.latent_loss_tracker.result(),
            'train_kl_loss': self.kl_loss_tracker.result()
        }

    @tf.function
    def test_step(self, inputs):
        x_data, y_data = inputs
        z_mean, z_log_var, z_sampling, z_obs = self.encoder(x_data)
        reco = self.decoder([z_sampling, z_obs])
        val_reco_loss = tf.reduce_mean(losses.mean_squared_error(x_data, reco))
        val_latent_loss = tf.reduce_mean(losses.mean_squared_error(y_data, z_obs))
        val_kl_loss = tf.reduce_mean(-0.5 * (1. + z_log_var - tf.square(z_mean)
                                             - tf.exp(z_log_var)))
        val_total_loss = val_reco_loss + self.beta*val_kl_loss + val_latent_loss

        self.val_loss_tracker.update_state(val_total_loss)
        self.val_reco_loss_tracker.update_state(val_reco_loss)
        self.val_latent_loss_tracker.update_state(val_latent_loss)
        self.val_kl_loss_tracker.update_state(val_kl_loss)

        return {
            'total_loss': self.val_loss_tracker.result(),
            'reco_loss': self.val_reco_loss_tracker.result(),
            'latent_loss': self.val_latent_loss_tracker.result(),
            'kl_loss': self.val_kl_loss_tracker.result()
        }

    @tf.function
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded[2:])
        return decoded


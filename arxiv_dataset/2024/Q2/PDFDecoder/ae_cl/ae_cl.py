import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, models
from tensorflow.keras.models import Model


class AutoencoderConstrainedLatent(Model):

    def __init__(self, latent_dim):
        super(AutoencoderConstrainedLatent, self).__init__()

        # Size of what's going into the autoencoder
        self.input_dim = 28*28

        # Size of the latent dimension
        self.latent_dim = latent_dim

        # Size of the observable dimension
        self.observable_dim = (self.latent_dim,)

        # Build encoder and decoder networks and model
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        # Track training losses in network
        self.total_loss_tracker = tf.keras.metrics.Mean(name='train_total_loss')
        self.reco_loss_tracker = tf.keras.metrics.Mean(name='train_reco_loss')
        self.latent_loss_tracker = tf.keras.metrics.Mean(name='train_latent_loss')

        # Track validation losses in network
        self.val_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
        self.val_reco_loss_tracker = tf.keras.metrics.Mean(name='reco_loss')
        self.val_latent_loss_tracker = tf.keras.metrics.Mean(name='latent_loss')

    def build_encoder(self):
        encoder_inputs = layers.Input(shape=(self.input_dim,))

        dense1 = layers.Dense(units=512, activation='ELU')(encoder_inputs)
        dense2 = layers.Dense(units=256, activation='ELU')(dense1)
        dense3 = layers.Dense(units=128, activation='ELU')(dense2)
        dense4 = layers.Dense(units=64, activation='ELU')(dense3)
        dense5 = layers.Dense(units=32, activation='ELU')(dense4)

        encoder_outputs = layers.Dense(units=self.latent_dim, activation='linear')(dense5)

        encoder_model = models.Model(inputs=encoder_inputs,
                                     outputs=encoder_outputs,
                                     name='encoder')

        return encoder_model

    def build_decoder(self):
        decoder_inputs = layers.Input(shape=(self.latent_dim,))

        dense1 = layers.Dense(units=32, activation='ELU')(decoder_inputs)
        dense2 = layers.Dense(units=64, activation='ELU')(dense1)
        dense3 = layers.Dense(units=128, activation='ELU')(dense2)
        dense4 = layers.Dense(units=256, activation='ELU')(dense3)
        dense5 = layers.Dense(units=512, activation='ELU')(dense4)

        decoder_outputs = layers.Dense(units=self.input_dim, activation='linear')(dense5)

        decoder_model = models.Model(inputs=decoder_inputs,
                                     outputs=decoder_outputs,
                                     name='decoder')

        return decoder_model

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reco_loss_tracker,
                self.latent_loss_tracker,
                self.val_loss_tracker,
                self.val_reco_loss_tracker,
                self.val_latent_loss_tracker]

    @tf.function
    def train_step(self, inputs):
        x_data, y_data = inputs
        with tf.GradientTape() as tape:
            z = self.encoder(x_data, training=True)
            reco = self.decoder(z, training=True)
            reco_loss = tf.reduce_mean(losses.mean_squared_error(x_data, reco))
            latent_loss = tf.reduce_mean(losses.mean_squared_error(y_data, z))
            total_loss = reco_loss + latent_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reco_loss_tracker.update_state(reco_loss)
        self.latent_loss_tracker.update_state(latent_loss)

        return {
            'train_total_loss': self.total_loss_tracker.result(),
            'train_reco_loss': self.reco_loss_tracker.result(),
            'train_latent_loss': self.latent_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, inputs):
        x_data, y_data = inputs
        z = self.encoder(x_data)
        reco = self.decoder(z)

        val_reco_loss = tf.reduce_mean(losses.mean_squared_error(x_data, reco))
        val_latent_loss = tf.reduce_mean(losses.mean_squared_error(y_data, z))

        val_total_loss = val_reco_loss + val_latent_loss

        self.val_loss_tracker.update_state(val_total_loss)
        self.val_reco_loss_tracker.update_state(val_reco_loss)
        self.val_latent_loss_tracker.update_state(val_latent_loss)

        return {
            'total_loss': self.val_loss_tracker.result(),
            'reco_loss': self.val_reco_loss_tracker.result(),
            'latent_loss': self.val_latent_loss_tracker.result(),
        }

    @tf.function
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# VRNN with disentangled representational learning + DAT (on z)
from tensorflow.keras.layers import Dense, Input, GRU, LSTM
from tensorflow.keras import Model, Sequential
from tensorflow.keras.activations import softplus
from tensorflow.keras import backend as K
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow import ones_like, zeros_like
import tensorflow as tf
import numpy as np
import math

class VRNN_DAT(Model):
    def __init__(self, x_dim, h_dim, z_dim, phi_x_dim=32, d_dim=128, vrnn_lr= 1e-3, adv_lr= 1e-3, clip=5, k=100, rnn="GRU"):
        super(VRNN_DAT, self).__init__()
        print("DAT on Zt")
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.rnn_backend = rnn

        self.clip = clip

        #feature-extracting transformations
        self.phi_x = Sequential([
            Dense(phi_x_dim, activation='relu'),
            Dense(phi_x_dim, activation='relu', name="phi_x")])

        self.phi_z = Sequential([
            Dense(z_dim, activation='relu', name="phi_z")])

        #encoder
        self.enc = Sequential([
            Dense(h_dim, activation='relu'),
            Dense(h_dim, activation='relu', name="enc")])
        self.enc_mean = Dense(z_dim, name="enc_mean")
        self.enc_std = Dense(z_dim, activation=softplus, name="enc_std")

        #prior
        self.prior = Dense(h_dim, activation='relu', name="prior")
        self.prior_mean = Dense(z_dim, name="prior_mean")
        self.prior_std = Dense(z_dim, activation=softplus, name="prior_std")

        #decoder
        self.dec = Sequential([
            Dense(h_dim, activation='relu'),
            Dense(h_dim, activation='relu', name="dec")])
        self.dec_std = Dense(x_dim, activation=softplus, name="dec_std") # No need for MNIST
        self.dec_mean = Dense(x_dim, name='dec_mean')
        
        # recurrence
        if rnn == "GRU":
            print("Using GRU Backend")
            self.rnn = GRU(h_dim, return_state=True)
        elif rnn == "LSTM":
            print("Using LSTM Backend")
            self.rnn = LSTM(h_dim, return_state=True)
            

        # DAT 
        self.discriminator = Sequential([
            GRU(d_dim),
            Dense(d_dim, activation='relu'),
            Dense(1, activation='softmax', name="discriminator")])
        self._bce = BinaryCrossentropy(name='BCE')
        self.k = k # constant to multiply with bce loss
        
        # train metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.rmse_metric = tf.keras.metrics.RootMeanSquaredError(name="rmse")

        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=adv_lr)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=vrnn_lr)
      
    @tf.function
    def train_step(self, data):
        x_train, y_train, domain_labels = data
        with tf.GradientTape() as VRNN_tape, tf.GradientTape() as ADV_tape:
            y_pred, phi_x, (all_enc_mean, all_enc_std), (all_dec_mean, all_dec_std), (all_pri_mean, all_pri_std)  = self.__call__(x_train, y_train, domain_labels)
            domain_labels = tf.convert_to_tensor(domain_labels)

            # computing losses
            # KLD
            kld_loss = self._kld_gauss(all_enc_mean, all_enc_std, all_pri_mean, all_pri_std)

            # Prior NLL
            # Supervise prior with ground truth, if present
            _prior_nll_loss = self._nll_gauss(all_pri_mean, all_pri_std, y_train * domain_labels)
            prior_nll_loss = _prior_nll_loss 

            # reconsruction loss
            reconstruction_loss = self._nll_gauss(all_dec_mean, all_dec_std, x_train)
            
            # DAT loss | y_pred = z_t
            d_pred = self.discriminator(y_pred)
            d_loss = self._bce(domain_labels, d_pred)       
            e_loss = self._bce(tf.ones_like(d_pred), d_pred)
            
            loss = tf.math.reduce_mean(kld_loss + reconstruction_loss + prior_nll_loss + self.k*e_loss)

        # Gradient descent vrnn
        vrnn_grads = VRNN_tape.gradient(loss, self.trainable_weights)
        # grad norm clipping
        vrnn_grads, _ = tf.clip_by_global_norm(vrnn_grads, self.clip)
        # optimize
        self.optimizer.apply_gradients(zip(vrnn_grads, self.trainable_weights))

        # Gradient descent adversarial
        adv_grads = ADV_tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(adv_grads, self.discriminator.trainable_weights))

        # Update metrics
        # return {loss, (kld_loss, reconstruction_loss, prior_nll_loss), (d_loss, e_loss)}
        self.loss_tracker.update_state(loss)
        self.rmse_metric.update_state(y_train, y_pred)
        return {"loss": self.loss_tracker.result(), "rmse": self.rmse_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.rmse_metric]
    
    def __call__(self, x, y, domain_labels):
        # x = (bs, seq_len, features) | y = (bs, labels) | class_labels = (bs, 1)
        domain_labels =  tf.convert_to_tensor(domain_labels)

        all_enc_mean, all_enc_std = [], []
        all_pri_mean, all_pri_std = [], []
        all_dec_mean, all_dec_std = [], []
        
        phi_x = []
        y_pred = []

        #  h0 (bs, h_dim)
        h = tf.zeros((tf.shape(x)[0], self.h_dim))
        
        if self.rnn_backend == "LSTM":
            c = tf.zeros((tf.shape(x)[0], self.h_dim))
        
        
        # for each timestep
        for t in range(x.shape[1]):

            # phi_x_t (bs, phi_x_dim)
            phi_x_t = self.phi_x(x[:, t, :])

            #encoder
            enc_t = self.enc(tf.concat([phi_x_t, h], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t) 

            #prior
            prior_t = self.prior(h)
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t, tf.shape(x)[0])
            phi_z_t = self.phi_z(z_t)
            # z_sample.append(z_t)

            #decoder
            dec_t = self.dec(tf.concat([phi_z_t, h], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)
            
            #recurrence
            if self.rnn_backend == "GRU":
                _, h = self.rnn(tf.expand_dims(tf.concat([phi_x_t, phi_z_t], 1), 1), initial_state=h)
            elif self.rnn_backend == "LSTM":
                 _, h, c = self.rnn(tf.expand_dims(tf.concat([phi_x_t, phi_z_t], 1), 1), initial_state=[h, c])
            else:
                raise("No known rnn backend")

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_pri_mean.append(prior_mean_t)
            all_pri_std.append(prior_std_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)
            phi_x.append(phi_x_t)
            y_pred.append(z_t)

        # z_sample = tf.transpose(tf.convert_to_tensor(z_sample), [1, 0, 2])
        all_enc_mean = tf.transpose(tf.convert_to_tensor(all_enc_mean), [1, 0, 2])
        all_enc_std = tf.transpose(tf.convert_to_tensor(all_enc_std), [1, 0, 2])

        all_dec_mean = tf.transpose(tf.convert_to_tensor(all_dec_mean), [1, 0, 2])
        all_dec_std = tf.transpose(tf.convert_to_tensor(all_dec_std), [1, 0, 2])

        all_pri_mean = tf.transpose(tf.convert_to_tensor(all_pri_mean), [1, 0, 2])
        all_pri_std = tf.transpose(tf.convert_to_tensor(all_pri_std), [1, 0, 2])
        
        phi_x = tf.transpose(tf.convert_to_tensor(phi_x), [1, 0, 2])
        y_pred = tf.transpose(tf.convert_to_tensor(y_pred), [1, 0, 2])

        return y_pred, phi_x, \
            (all_enc_mean, all_enc_std), \
            (all_dec_mean, all_dec_std), \
            (all_pri_mean, all_pri_std) 

    def _reparameterized_sample(self, mean, std, bs):
        """using std to sample"""
        eps = K.random_normal((bs, self.z_dim), 0.0, 1.0, dtype=tf.float32) 
        return eps*std + mean
        # return eps.mul(std).add_(mean)
        # return z_mean + K.exp(0.5 * z_log_sigma) * epsilon

    def DAT_loss(self, class_labels, y_pred):
        return self._bce(y_true=class_labels, y_pred=y_pred)

    

    def _kld_gauss(self, mu_1, sigma_1, mu_2, sigma_2):
        return tf.math.reduce_sum(0.5 * (
            2 * K.log(K.maximum(1e-9,sigma_2)) 
            - 2 * K.log(K.maximum(1e-9,sigma_1))
            + (K.square(sigma_1) + K.square(mu_1 - mu_2)) / K.maximum(1e-9,(K.square(sigma_2))) - 1
        ))

    def _nll_gauss(self, mean, std, x):
        ss = tf.math.maximum(1e-10, tf.square(std))
        norm = tf.math.subtract(x, mean)
        z = tf.math.divide(tf.math.square(norm), ss)
        denom_log = tf.math.log(2*np.pi*ss, name='denom_log')

        result = tf.math.reduce_sum(z+denom_log)/2
        return result 
    
    def get_feature_representation(self, x):
        predicted = []
        
        seq_len = x.shape[1]
        h = tf.zeros((x.shape[0], self.h_dim))
        
        if self.rnn_backend == "LSTM":
            c = tf.zeros((tf.shape(x)[0], self.h_dim))
            
        for t in range(seq_len):
            # phi_x_t (bs, h_dim)
            phi_x_t = self.phi_x(x[:, t, :])

            #encoder
            enc_t = self.enc(tf.concat([phi_x_t, h], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t) 

            #sampling and reparameterization
            z_t = enc_mean_t
            phi_z_t = self.phi_z(z_t)

            #recurrence
            if self.rnn_backend == "GRU":
                _, h = self.rnn(tf.expand_dims(tf.concat([phi_x_t, phi_z_t], 1), 1), initial_state=h)
            elif self.rnn_backend == "LSTM":
                 _, h, c = self.rnn(tf.expand_dims(tf.concat([phi_x_t, phi_z_t], 1), 1), initial_state=[h, c])
            else:
                raise("No known rnn backend")

            predicted.append(phi_x_t)
        return np.swapaxes(np.array(predicted), 0, 1)    
    
    def predict(self, x, return_phi_z=False):
        predicted = []
        
        seq_len = x.shape[1]
        h = tf.zeros((x.shape[0], self.h_dim))
        if self.rnn_backend == "LSTM":
            c = tf.zeros((tf.shape(x)[0], self.h_dim))
            
        for t in range(seq_len):
            # phi_x_t (bs, h_dim)
            phi_x_t = self.phi_x(x[:, t, :])

            #encoder
            enc_t = self.enc(tf.concat([phi_x_t, h], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t) 

            #sampling and reparameterization
            # z_t = self._reparameterized_sample(enc_mean_t, enc_std_t, x.shape[0])
            z_t = enc_mean_t
            phi_z_t = self.phi_z(z_t)

            #recurrence
            if self.rnn_backend == "GRU":
                _, h = self.rnn(tf.expand_dims(tf.concat([phi_x_t, phi_z_t], 1), 1), initial_state=h)
            elif self.rnn_backend == "LSTM":
                 _, h, c = self.rnn(tf.expand_dims(tf.concat([phi_x_t, phi_z_t], 1), 1), initial_state=[h, c])
            else:
                raise("No known rnn backend")

            if return_phi_z:
                predicted.append(phi_z_t)    
            else:
                predicted.append(z_t)
        return np.swapaxes(np.array(predicted), 0, 1)    


if __name__== "__main__":
    vrnn = VRNN(6, 20, 93, 128) 
    d_dim = 128
    mask =  np.random.randint(2, size=32).astype(np.float32)
    # mask = np.zeros_like((32)).astype(np.float32)
    adversarial = Discriminator(d_dim)
    adv_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    z_samples, (all_enc_mean, all_enc_std), (all_dec_mean, all_dec_std), (all_pri_mean, all_pri_std) = vrnn(np.random.normal(0, 1, size=(32, 50, 6)), np.random.normal(0, 1, size=(32, 50, 93)), mask, adversarial )
    # print(kld_loss, nll_loss, prior_nll_loss)
    # pred = vrnn.predict(np.random.normal(0, 1, size=(32, 50, 6)))
    # print(pred.shape)
   
    print(all_pri_mean.shape)

    vrnn.train_step(np.random.normal(0, 1, size=(32, 50, 6)).astype(np.float32), np.random.normal(0, 1, size=(32, 50, 93)).astype(np.float32), mask, adversarial, adv_optimizer, step=1)

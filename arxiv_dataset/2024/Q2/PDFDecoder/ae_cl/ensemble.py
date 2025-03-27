# Import system libraries for path dependence
import os
import sys
import pathlib

# Import numerical libraries
import numpy as np

# Import tensorflow libraries
from tensorflow.keras import layers, optimizers, losses, models
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import History

# Import the model
from ae_cl import AutoencoderConstrainedLatent

# Set up directories for storing model information
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
model_dir = parent_dir + '/trained_models'
history_dir = parent_dir + '/model_history'
output_dir = parent_dir + '/output_data'
sys.path.insert(1, model_dir)
sys.path.insert(2, history_dir)
sys.path.insert(3, output_dir)


# Ensemble models together
class Ensemble:

    def __init__(self, latent_dim):

        # Model specific variables
        self.latent_dim = latent_dim
        self.model = AutoencoderConstrainedLatent(self.latent_dim)
        self.optimizer = optimizers.legacy.Adam(learning_rate=5e-5)
        self.batch_size = 32
        self.num_epochs = 10_000

        # Directory variables
        new_model_dir_name = f'ae_cl_l{self.latent_dim}_models'
        new_history_dir_name = f'ae_cl_l{self.latent_dim}_history'
        new_output_dir_name = f'ae_cl_l{self.latent_dim}_outputs'
        self.new_model_dir = pathlib.Path(model_dir, new_model_dir_name)
        self.new_history_dir = pathlib.Path(history_dir, new_history_dir_name)
        self.new_output_dir = pathlib.Path(output_dir, new_output_dir_name)
        self.new_model_dir.mkdir(parents=True, exist_ok=True)
        self.new_history_dir.mkdir(parents=True, exist_ok=True)
        self.new_output_dir.mkdir(parents=True, exist_ok=True)

        # Ensemble of encoded variables and decoded outputs
        self.encoded_obs = []
        self.decoded_outputs = []
        self.z_space = []
        self.history = History()

    def train(self, x, y, num_runs):

        # Separate train and validation sets
        x_train, x_valid = x
        y_train, y_valid = y

        for run in range(num_runs):

            print('Run:',run)
            # Set up file paths
            model_filepath = self.new_model_dir / f'model_{run}.hdf5'
            history_filepath = self.new_history_dir / f'model_{run}.csv'

            # Initialize callback objects
            checkpoint = ModelCheckpoint(model_filepath,
                                         monitor='train_total_loss',
                                         verbose=0,
                                         save_best_only=True,
                                         save_weights_only=True,
                                         mode='auto',
                                         save_frequency=1)

            history_logger = CSVLogger(history_filepath,
                                       separator=',')

            reduce_lr = ReduceLROnPlateau(monitor='train_total_loss',
                                          factor=0.75,
                                          patience=10,
                                          min_delta=1e-6,
                                          cooldown=5,
                                          min_lr=1e-6)

            early_stop = EarlyStopping(monitor='val_total_loss', patience=25)

            # Initialize and compile model
            self.model = AutoencoderConstrainedLatent(self.latent_dim)
            self.model.compile(optimizer=self.optimizer)

            # Construct history object with fitted model
            self.history = self.model.fit(x=x_train,
                                          y=y_train,
                                          epochs=self.num_epochs,
                                          shuffle=True,
                                          verbose=0,
                                          batch_size=self.batch_size,
                                          validation_data=[x_valid, y_valid],
                                          callbacks=[checkpoint,
                                                     history_logger,
                                                     reduce_lr,
                                                     early_stop])
        return self.history

    def get_ensemble(self, x):

        # Re-initialize return variables so it can be re-run
        self.encoded_obs = []
        self.decoded_outputs = []

        # Iterate through the model directory
        for file in os.listdir(self.new_model_dir):
            # Build the model
            self.model.build(x.shape)

            # Load model from weights file
            self.model.load_weights(self.new_model_dir / file)

            # Encode x inputs and get outputs
            z = self.model.encoder(x)
            outputs = self.model.decoder(z).numpy()

            # Put together return variables
            self.encoded_obs.append(z)
            self.decoded_outputs.append(outputs)

        # Make return lists into arrays
        self.encoded_obs = np.array(self.encoded_obs)
        self.decoded_outputs = np.array(self.decoded_outputs)

        np.save(output_dir + f'/pdf_outputs_latent_{self.latent_dim}.npy',
                self.decoded_outputs)
        np.save(output_dir + f'/moments_encoded_latent_{self.latent_dim}.npy',
                self.encoded_obs)

        return self.encoded_obs, self.decoded_outputs

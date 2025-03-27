import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import argparse
import glob
import os
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
GENRES = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
N_FRAMES = 625

# Function to load data
def load_data(data_dir):
    X = []
    y = []
    genre_dirs = glob.glob(os.path.join(data_dir, '*'))
    for dir in genre_dirs:
        genre = os.path.basename(dir)
        file_paths = glob.glob(os.path.join(dir, '*.p'))
        for p in file_paths:
            with open(p, 'rb') as f:
                feature_data = pickle.load(f)
                start_idx = len(feature_data) // 2 - N_FRAMES // 2
                feature_segment = feature_data[start_idx:start_idx + N_FRAMES, :]
                X.append(feature_segment)
            label = GENRES.index(genre)
            y.append(tf.keras.utils.to_categorical(label, len(GENRES)))
    return np.array(X), np.array(y)

# Set up argument parser
parser = argparse.ArgumentParser(description='Load a DA model, predict genres, and plot a confusion matrix.')
parser.add_argument('--val-folder', type=str, required=True, help='Path to the folder containing validation data')
parser.add_argument('--da-model', type=str, required=True, help='Path to the DA model file')
args = parser.parse_args()

# Load the DA base model
da_base_model = tf.keras.models.load_model(args.da_model,
                                            custom_objects={"contrastive_loss": tfa.losses.contrastive_loss})
model_name = os.path.splitext(os.path.basename(args.da_model))[0]
model_in = da_base_model.layers[0].input
model_out = da_base_model.layers[-2].output
da_model = tf.keras.models.Model(model_in, model_out)

# Load the data to predict on
X_predict, y_predict = load_data(args.val_folder)
print(f">>>>>>>>>>>>>>>>>>>>>Shape of the X_predict is: {X_predict.shape}")

# Predict on the new data
y_pred_predict = da_model.predict(X_predict)
y_pred_predict_classes = np.argmax(y_pred_predict, axis=1)

# Convert one-hot encoded y_predict back to class indices for confusion matrix
y_predict_indices = np.argmax(y_predict, axis=1)

# Generate confusion matrix
cm_predict = confusion_matrix(y_predict_indices, y_pred_predict_classes)

# Plot and save the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_predict, annot=True, fmt='g', cmap='Blues', xticklabels=GENRES, yticklabels=GENRES)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Using DA')
plt.show()
save_path = f"./synth_confusion_matrices/confusion_matrix_{model_name}.png" 
plt.savefig(save_path)

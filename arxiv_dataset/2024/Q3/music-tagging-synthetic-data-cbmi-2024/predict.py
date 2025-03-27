import argparse
import numpy as np
import pandas as pd
import glob
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import tensorflow_addons as tfa

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
                # Load the entire 30-second feature data
                feature_data = pickle.load(f)
                # Calculate the starting index to extract the middle 10-second segment
                start_idx = len(feature_data) // 2 - N_FRAMES // 2
                # Extract the middle 10-second segment
                feature_segment = feature_data[start_idx:start_idx + N_FRAMES, :]
                X.append(feature_segment)
            label = GENRES.index(genre)
            y.append(tf.keras.utils.to_categorical(label, len(GENRES)))
    return np.array(X), np.array(y)

# Set up argument parser
parser = argparse.ArgumentParser(description='Predict genres using a trained feed-forward network.')
parser.add_argument('--val-folder', type=str, required=True, help='Path to the folder containing validation data')
parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
args = parser.parse_args()

# Specify the tfa.losses.contrastive_loss used for DA
contr_loss = {"contrastive_loss": tfa.losses.contrastive_loss}
# Load the trained model
model = tf.keras.models.load_model(args.model, custom_objects=contr_loss)
model_name = os.path.splitext(os.path.basename(args.model))[0]

# Load the data to predict on
X_predict, y_predict = load_data(args.val_folder)

# Predict on the new data
print(f">>>>>>>>>>>>>>>>>>>>>Shape of the X_predict is: {X_predict.shape}")
y_pred_predict = model.predict(X_predict)
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
plt.title('Confusion Matrix - Target Only')
plt.show()
save_path = f"./synth_confusion_matrices/confusion_matrix_{model_name}.png" 
plt.savefig(save_path)


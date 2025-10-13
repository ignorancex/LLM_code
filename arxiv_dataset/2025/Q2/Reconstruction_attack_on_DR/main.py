import numpy as np
from training_data import *
from recons_network import *
from sklearn.metrics import mean_squared_error
from config import img_size, n_compo

# Create data for reconstruction
begin_create_data()

# Load embedded data from .npy files
X = np.load('X_recon.npy')
y = np.load('y_recon.npy')

# Preprocess and split the data into training and testing sets
# process_data() handles normalization, shuffling, and splitting
X_train, X_test, y_train, y_test = process_data(X, y)

# Create a PyTorch Dataset object for training data
dataset = Data(X_train, y_train.reshape(-1, img_size*img_size))

# Split dataset into training (4000 samples) and validation (500 samples) sets
train_set, val_set = torch.utils.data.random_split(dataset, [4000, 500])

# Create DataLoader objects for batch processing
trainloader = DataLoader(train_set, batch_size=64, shuffle=True)
valloader = DataLoader(val_set, batch_size=64, shuffle=False)

# Initialize the neural network model
model = Net(X_train.shape[1] - n_compo).to(device)

# Train the model using the training and validation data
model = train_model(model, trainloader, valloader)

# Perform inference on the test set without tracking gradients
with torch.no_grad():
    predict = model(torch.tensor(X_test, dtype=torch.float32).to(device))

# Convert predictions to numpy array and reshape to match original image dimensions
predict9 = predict.cpu().detach().numpy().reshape(-1, img_size*img_size)

# Calculate Mean Squared Error between predictions and ground truth
mse = mean_squared_error(y_test.reshape(-1, img_size*img_size), predict9)

# Report model performance
print(f"Mean Squared Error : {mse:.6f}")

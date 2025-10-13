import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Define a linear model
class LinearClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def train_linear_classifier(features, targets, pos_weight=torch.tensor(1.0)):
    targets = targets.float()
    #print("max targets: {}".format(torch.max(targets)))
    #raise RuntimeError
    #print(features.dtype)
    #print(targets)
    # Initialize the model, loss function, and optimizer
    input_dim = features.shape[1]
    model = LinearClassifier(input_dim).to("cuda")
    criterion = nn.BCEWithLogitsLoss()  # This combines a sigmoid layer and binary cross-entropy loss
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(features).squeeze()  # Shape (N,)
        #print(outputs.shape)
        #print(outputs.dtype)
        loss = F.binary_cross_entropy_with_logits(outputs, targets, pos_weight=pos_weight) #criterion(outputs, targets.long())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        """
        if (epoch + 1) % 10 == 0:
            #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            outputs = torch.where(outputs > 0, 1.0, 0.0)
            outputs = torch.where(outputs == targets, 1.0, 0.0)
            print("accuracy: {}".format(torch.mean(outputs)))
        """

    outputs = torch.where(outputs > 0, 1.0, 0.0)
    outputs = torch.where(outputs == targets, 1.0, 0.0)
    accuracy = torch.mean(outputs)

    return model.linear.weight.detach().squeeze(), model.linear.bias.detach().squeeze(), accuracy
    # After training, you can make predictions with:
    #with torch.no_grad():
    #    predictions = torch.sigmoid(model(features)).squeeze()
    #    predicted_classes = (predictions > 0.5).int()  # Convert probabilities to binary classes

    #print("Predicted classes:", predicted_classes)
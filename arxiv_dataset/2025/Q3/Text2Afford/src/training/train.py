import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm

# Define a simple binary classification model
# class SimpleClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         return x

# Define training function
def train_model(model, train_dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_dataloader, desc="Training"):

        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device, torch.float16)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=labels)
        
        loss = outputs.loss

        # print(input_ids)
        # print(pixel_values)
        print("Loss:", loss.item())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # inputs, labels = batch
        # inputs, labels = inputs.to(device), labels.to(device)

        # optimizer.zero_grad()
        # outputs = model(inputs)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    return average_loss

# Define validation function
def evaluate_model(model, val_dataloader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device, torch.float16)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=labels)
            
            loss = outputs.loss

            total_loss += loss.item()

    average_loss = total_loss / len(val_dataloader)
    return average_loss

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Dummy DataLoader for illustration purposes (replace with your actual DataLoader)
# # Assume input_size is the size of your input features, and output_size is 1 for binary classification
# dummy_train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
# dummy_val_dataloader = DataLoader(dataset_val, batch_size=64, shuffle=False)

# # Initialize model, criterion, and optimizer
# model = SimpleClassifier(input_size, hidden_size, output_size).to(device)
# criterion = nn.BCELoss()
# optimizer = AdamW(model.parameters(), lr=1e-3)

# # Training loop
# num_epochs = 5
# for epoch in range(num_epochs):
#     print(f"Epoch {epoch + 1}/{num_epochs}")
#     avg_train_loss = train_model(model, dummy_train_dataloader, optimizer, criterion, device)
#     print(f"Average Training Loss: {avg_train_loss}")

#     # Validation
#     avg_val_loss = evaluate_model(model, dummy_val_dataloader, criterion, device)
#     print(f"Average Validation Loss: {avg_val_loss}")

# # Save the trained model if needed
# torch.save(model.state_dict(), "trained_model.pth")

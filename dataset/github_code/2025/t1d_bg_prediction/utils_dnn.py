import time
import numpy as np
import torch
from torch import nn
from torch.utils import data
from copy import deepcopy


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout, activation):
        super().__init__()

        if activation == "relu":
            act_func = nn.ReLU()
        elif activation == "tanh":
            act_func = nn.Tanh()
        else:
            raise

        layers = []
        in_features = input_size

        # Add hidden layer
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(act_func)
            layers.append(nn.Dropout(dropout))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, input_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class RNNModel(nn.Module):
    def __init__(self, variant, input_size, hidden_size, num_layers, dropout):
        super().__init__()

        if variant == "lstm":
            self.rnn = nn.LSTM
        elif variant == "gru":
            self.rnn = nn.GRU
        else:
            raise

        self.rnn = self.rnn(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 0 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def get_params_count(model):
    no_params = sum(p.numel() for p in model.parameters())
    no_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {no_params}")
    print(f"Trainable params: {no_trainable_params}")


def get_dataloader(train_set, test_set, X_cols, y_cols, model_type, batch_size):
    assert model_type in ["mlp", "lstm", "gru"], "please choose from [mlp/lstm/gru]"

    X_train = torch.tensor(train_set[X_cols].values, dtype=torch.float32)
    y_train = torch.tensor(train_set[y_cols].values, dtype=torch.float32)

    X_test = torch.tensor(test_set[X_cols].values, dtype=torch.float32)
    y_test = torch.tensor(test_set[y_cols].values, dtype=torch.float32)

    if model_type != "mlp":
        X_train, y_train = X_train.unsqueeze(2), y_train.unsqueeze(2)
        X_test, y_test = X_test.unsqueeze(2), y_test.unsqueeze(2)

    train_loader = data.DataLoader(
        data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size
    )
    test_loader = data.DataLoader(
        data.TensorDataset(X_test, y_test), shuffle=False, batch_size=batch_size
    )

    return train_loader, test_loader


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    early_stopper,
    train_loader,
    test_loader,
    device,
    epochs,
    output_path,
    exp_name,
):
    best_loss = np.inf
    best_epoch = 0
    best_model = deepcopy(model)

    for epoch in range(epochs):
        # Train
        t = time.time()
        model.to(device)
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(targets, outputs)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(targets, outputs)
                val_loss += loss.item() * inputs.size(0)

        print(
            f"Epoch: {epoch}\tEval loss: {val_loss:.2f}\tElapsed time: {time.time() - t:.2f}\tLR: {scheduler.get_last_lr()[0]}"
        )

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model = deepcopy(model)
            torch.save(model.to("cpu").state_dict(), f"{output_path}/{exp_name}.pt")

        if early_stopper.early_stop(val_loss):
            break

    print(f"Best loss: {best_loss:.4f} @ epoch {best_epoch}")

    return best_model


@torch.no_grad()
def predict_in_batches(model, data, device, model_type="mlp", batch_size=256):
    assert model_type in ["mlp", "lstm", "gru"], "please choose from [mlp/lstm/gru]"

    model.to(device)
    num_samples = len(data)
    num_batches = int(np.ceil(num_samples / batch_size))
    y_preds = np.zeros((num_samples, 1))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        inputs = torch.tensor(data[start_idx:end_idx].values, dtype=torch.float32)
        inputs = inputs if model_type == "mlp" else inputs.unsqueeze(2)
        outputs = model(inputs.to(device))
        outputs = (
            outputs[:, -1].unsqueeze(1) if model_type == "mlp" else outputs[:, -1, :]
        )
        y_preds[start_idx:end_idx] = outputs.cpu().numpy()

    return y_preds

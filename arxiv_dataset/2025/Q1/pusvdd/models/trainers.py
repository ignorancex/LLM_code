import torch
from torch.utils.data import DataLoader

from models import detectors, losses


class Trainer:
    def __init__(self, device: torch.device):
        self.device = device
        self.train_losses: list[float] = []
        self.valid_losses: list[float] = []
        self.min_valid_loss = float("inf")

    def fit(
        self,
        model: detectors.Detector,
        criterion: losses.Loss,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        checkpoint: str,
        n_epoch: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
    ):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for epoch in range(n_epoch):
            # training step
            model.train()
            mean_train_loss: float = 0

            for batch in train_loader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)

                optimizer.zero_grad()
                output = model(x)
                train_loss = criterion(output, y)
                train_loss.backward()
                optimizer.step()

                mean_train_loss += train_loss.item() / len(train_loader)

            self.train_losses.append(mean_train_loss)

            prompt = f"epoch: {epoch} / Train: {mean_train_loss:.3f}"

            # validation step
            model.eval()
            mean_valid_loss: float = 0
            with torch.no_grad():
                for batch in valid_loader:
                    x = batch[0].to(self.device)
                    y = batch[1].to(self.device)

                    output = model(x)
                    valid_loss = criterion(output, y)
                    mean_valid_loss += valid_loss.item() / len(valid_loader)

            self.valid_losses.append(mean_valid_loss)
            prompt += f" / Valid: {mean_valid_loss:.3f}"

            # Early stopping
            if mean_valid_loss < self.min_valid_loss:
                self.min_valid_loss = mean_valid_loss
                torch.save(model.state_dict(), checkpoint)
                prompt += " / Save"

            print(prompt)

        model.load_state_dict(torch.load(checkpoint, weights_only=True))

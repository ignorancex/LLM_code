import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter:
    def __init__(self):
        self.count = 0
        self.sum: torch.Tensor | float = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> torch.Tensor | float:
        return self.sum / self.count if self.count > 0 else 0.0


def accuracy(logits, labels):
    predicted_labels = torch.argmax(logits, dim=1)
    labels = labels.long()
    return torch.mean((predicted_labels == labels).float())


def train_one_epoch(model, optimizer, train_loader):
    model.train()

    non_blocking = True if torch.cuda.is_available() else False
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE, non_blocking=non_blocking), targets.to(DEVICE, non_blocking=non_blocking)
        optimizer.roll(compute_cmp_state_kwargs={"model": model, "inputs": inputs, "targets": targets})


@torch.inference_mode()
def validate_one_epoch(model, cmp, loader):
    model.eval()

    accuracies = AverageMeter()

    non_blocking = True if torch.cuda.is_available() else False
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE, non_blocking=non_blocking), targets.to(DEVICE, non_blocking=non_blocking)
        logits = model(inputs)
        accuracies.update(accuracy(logits, targets), inputs.shape[0])

    model_density, _ = cmp.compute_sparsity_stats(model, is_test_time=True)

    return model_density, accuracies.avg

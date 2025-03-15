import torch
from torch.utils.data import DataLoader


def show_dataloader_info(dataloader: DataLoader) -> None:
    """
    Show information about the dataloader.

    Args:
        dataloader (DataLoader): The dataloader to show information about.

    Returns:
        None
    """
    num_labels: int = 0
    labels = set()
    for _, label in dataloader:
        label: torch.Tensor
        labels.update(label.tolist())
    num_labels = len(labels)

    print(f"\n---- Dataloader Info ----")
    print(f"Dataloader Length: {len(dataloader)}")
    print(f"Dataset Size: {len(dataloader.dataset)}")
    print(f"Batch Size: {dataloader.batch_size}")
    print(f"Sample Size: {dataloader.dataset[0][0].shape}")
    print(f"Num Labels: {num_labels}")
    print(f"---- End of Dataloader Info ----\n")

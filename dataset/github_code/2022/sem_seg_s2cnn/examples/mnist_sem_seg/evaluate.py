import numpy as np

from tabulate import tabulate
import torch
import torch.nn as nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_metrics(segmenter, data_loader):
    criterion = nn.CrossEntropyLoss(reduction="sum")
    batch_size = data_loader.batch_size
    n_batches = len(data_loader)

    confusion_matrix = np.zeros((11, 11))
    correct = 0
    n_pixels = 0
    total_loss = 0.0

    for batch, (images, labels) in enumerate(data_loader):

        segmenter.eval()

        images = images.to(DEVICE)
        labels = labels.long().to(DEVICE)

        batch_slice = slice(batch * batch_size, batch * batch_size + len(images))

        with torch.no_grad():
            outputs = segmenter(images)

        _, predicted = torch.max(outputs, 1)
        n_pixels += images.numel()
        correct += (predicted == labels).long().sum()

        for true in range(11):
            for pred in range(11):
                confusion_matrix[pred, true] += (
                    torch.logical_and(labels == true, predicted == pred).long().sum().cpu()
                )

        total_loss += criterion(outputs, labels)

        if n_batches > 5 and batch % (n_batches // 5) == 0 and batch != 0:
            print(f"Batch {batch + 1}/{n_batches} done")

    loss = total_loss / n_pixels

    result = {}

    result["loss"] = loss.item()

    # note: this is not quite the same as 100 * total_correct / total_total
    result["mean_accuracy"] = 100 * correct / n_pixels

    confusion_matrix /= np.sum(confusion_matrix)
    confusion_matrix *= 100

    non_bkgd_cm = confusion_matrix[1:, 1:].copy()
    non_bkgd_cm /= np.sum(non_bkgd_cm)
    non_bkgd_cm *= 100

    result["non_bkgd_accuracy"] = non_bkgd_cm.trace()

    for i in range(11):
        intersection = confusion_matrix[i, i]
        union = np.sum(confusion_matrix[i]) + np.sum(confusion_matrix[:, i]) - intersection
        result[f"iou_class_{i}"] = 100 * intersection / union

    result["iou"] = np.mean([result[f"iou_class_{i}"] for i in range(11)])
    result["non_bkgd_iou"] = np.mean([result[f"iou_class_{i}"] for i in range(1, 11)])

    confusion_matrix = confusion_matrix.tolist()
    confusion_matrix.insert(
        0,
        [
            "bkgd",
            "class 0",
            "class 1",
            "class 2",
            "class 3",
            "class 4",
            "class 5",
            "class 6",
            "class 7",
            "class 8",
            "class 9",
        ],
    )
    confusion_matrix[0].insert(0, "prediction \\ target")
    confusion_matrix[1].insert(0, "bkgd")
    for i in range(10):
        confusion_matrix[i + 2].insert(0, "class " + str(i))

    table = tabulate(confusion_matrix, headers="firstrow", floatfmt=".3f", tablefmt="grid")

    result["confusion_matrix"] = table

    return result

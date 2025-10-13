import torch


def collate_fn(dataset_items: list[dict]):
    result_batch = {}

    for column_name in dataset_items[0].keys():
        if not isinstance(dataset_items[0][column_name], torch.Tensor):
            continue
        result_batch[column_name] = torch.vstack(
            [elem[column_name] for elem in dataset_items if elem]
        )

    return result_batch

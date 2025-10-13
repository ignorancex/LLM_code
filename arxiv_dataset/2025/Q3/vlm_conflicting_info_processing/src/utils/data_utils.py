from torch.utils.data import DataLoader

from data.multimodal_torchvision import MultimodalDataset, LmEvaluationDataCollator

def get_dataset(split, prompt_generator, args):
    if args.dataset in ["cifar10", "cifar100"]:
        data_wrapper = MultimodalDataset
    elif args.dataset in ["imagenet", "imagenet100", "Pascal", "CUB_color"]:
        data_wrapper = MultimodalDataset
        if split == "test":
            split="val"
    else:
        raise NotImplementedError(f"{args.dataset} not supported yet.")

    dataset = data_wrapper(args, split, prompt_generator)
    print(f"dataset_{split}: {len(dataset)}")
    return dataset


def get_dataloader(dataset, processor, args, is_train):
    
    if args.dataset in ["cifar10", "cifar100", "imagenet100", "Pascal", "CUB_color"]:
        collator = LmEvaluationDataCollator(processor, False, text_only=(args.caption_type=="text_only"))
    else:
        raise NotImplementedError(f"{args.dataset} not supported yet.")

    return DataLoader(
        dataset, 
        batch_size=args.batch_size,
        collate_fn=collator,
        pin_memory=False,
        shuffle=True if is_train else False,
        drop_last=True if is_train else False,
        num_workers=0
    )

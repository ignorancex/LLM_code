import os
from functools import partial
from collections import OrderedDict

import torch

import torch_em
from torch_em.data import datasets
from torch_em.data import MinTwoInstanceSampler, ConcatDataset

import micro_sam.training as sam_training
from micro_sam.instance_segmentation import get_unetr

from patho_sam.training import SemanticInstanceTrainer, get_train_val_split
from patho_sam.training.util import remap_labels, calculate_class_weights_for_loss_weighting


def get_dataloaders(patch_shape, data_path):
    """This returns the data loaders implemented in `torch-em`.
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/histopathology/
    It will automatically download all the histopathology datasets used here.

    NOTE: To replace this with another data loader, you need to return a torch data loader
    that returns `x, y` tensors, where `x` is the image data and `y` are corresponding labels.
    The labels have to be in a label mask semantic segmentation format.
    i.e. a tensor of the same spatial shape as `x`, with semantic labels for objects.
    Important: the ID 0 is reserved for background and ensure you have all semantic classes.
    """
    raw_transform = sam_training.identity
    sampler = MinTwoInstanceSampler()
    label_dtype = torch.float32

    # PanNuke dataset
    pannuke_ds = datasets.get_pannuke_dataset(
        path=os.path.join(data_path, "pannuke"),
        patch_shape=(1, *patch_shape),
        ndim=2,
        folds=["fold_1", "fold_2"],
        custom_label_choice="semantic",
        sampler=sampler,
        label_dtype=label_dtype,
        raw_transform=raw_transform,
        download=True,
    )

    # PUMA dataset.
    def _get_puma_ds(split):
        return datasets.get_puma_dataset(
            path=os.path.join(data_path, "puma"),
            patch_shape=patch_shape,
            split=split,
            label_choice="semantic",
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=partial(remap_labels, name="puma"),
            raw_transform=raw_transform,
            download=True,
        )

    # Create custom splits for PanNuke.
    pannuke_train_ds, pannuke_val_ds = get_train_val_split(ds=pannuke_ds)

    # Create a concatenation of all datasets.
    _train_datasets = [_get_puma_ds("train"), pannuke_train_ds]
    train_ds = ConcatDataset(*_train_datasets)

    _val_datasets = [_get_puma_ds("val"), pannuke_val_ds]
    val_ds = ConcatDataset(*_val_datasets)

    # Get the dataloaders.
    train_loader = torch_em.get_data_loader(train_ds, batch_size=8, shuffle=True, num_workers=16)
    val_loader = torch_em.get_data_loader(val_ds, batch_size=1, shuffle=True, num_workers=16)

    return train_loader, val_loader


def train_semantic_segmentation_generalist(args):
    """Code for semantic segmentation for multiple histopathology datasets.
    """
    # Hyperparameters for training
    model_type = args.model_type
    num_classes = 6  # available classes are [0, 1, 2, 3, 4, 5]
    checkpoint_path = args.checkpoint_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_name = f"{model_type}/generalist_semantic"

    train_loader, val_loader = get_dataloaders(patch_shape=(512, 512), data_path=args.input_path)

    # Whether we opt for finetuning decoder only or finetune the entire backbone.
    if args.decoder_only:
        freeze = ["image_encoder", "prompt_encoder", "mask_decoder"]
        checkpoint_name += "/finetune_decoder_only"
    else:
        freeze = None
        checkpoint_name += "/finetune_all"

    # Get the trainable Segment Anything Model.
    model, state = sam_training.get_trainable_sam_model(
        model_type=model_type, device=device, checkpoint_path=checkpoint_path, freeze=freeze, return_state=True,
    )

    # Whether to use the pretrained decoder (used for AIS) or train from scratch.
    if args.decoder_from_pretrained:
        decoder_state = []
        # Remove the output layer weights as we have new target class for the new task.
        decoder_state = OrderedDict(
            [(k, v) for k, v in state["decoder_state"].items() if not k.startswith("out_conv.")]
        )
        checkpoint_name += "-from_pretrained"
    else:
        decoder_state = None
        checkpoint_name += "-from_scratch"

    # Get the UNETR model for semantic segmentation pipeline
    unetr = get_unetr(
        image_encoder=model.sam.image_encoder,
        decoder_state=decoder_state,
        device=device,
        out_channels=num_classes,
        flexible_load_checkpoint=True,
    )

    # All other stuff we need for training
    optimizer = torch.optim.AdamW(unetr.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5)

    # This class creates all the training data for each batch (inputs and semantic labels)
    convert_inputs = sam_training.util.ConvertToSemanticSamInputs()

    # The trainer which performs the semantic segmentation training and validation (implemented using 'torch_em')
    trainer = SemanticInstanceTrainer(
        name=checkpoint_name,
        save_root=args.save_root,
        train_loader=train_loader,
        val_loader=val_loader,
        model=unetr,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        log_image_interval=10,
        mixed_precision=True,
        compile_model=False,
        convert_inputs=convert_inputs,
        num_classes=num_classes,
        dice_weight=0,
        class_weights=calculate_class_weights_for_loss_weighting(),
    )
    trainer.fit(iterations=int(args.iterations), overwrite_training=False)


def main(args):
    train_semantic_segmentation_generalist(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", default="/mnt/vast-nhr/projects/cidas/cca/test/data", type=str,
        help="Path where you would like to store the training data."
    )
    parser.add_argument(
        "-m", "--model_type", default="vit_b_histopathology", type=str,
        help="The choice of model to perform semantic segmentation on."
    )
    parser.add_argument(
        "-c", "--checkpoint_path", default=None, type=str,
        help="Filepath to the model with which you would like to perform downstream semantic segmentation."
    )
    parser.add_argument(
        "-s", "--save_root", default=None, type=str,
        help="Filepath where to store the trained model checkpoints and logs."
    )
    parser.add_argument(
        "--iterations", default=1e5, type=str,
        help="The total number of iterations to train the model for."
    )
    parser.add_argument(
        "--decoder_only", action="store_true",
        help="Whether to train the decoder only (by freezing the image encoder), or train all parts."
    )
    parser.add_argument(
        "--decoder_from_pretrained", action="store_true",
        help="Whether to train the decoder from scratch, or train the pretrained decoder (i.e. used for AIS)."
    )
    args = parser.parse_args()
    main(args)

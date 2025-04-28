"""MSP for OpenOOD benchmark"""
# Python
import argparse
import logging
import os
import sys
# Third Party
from jaxtyping import Float
import torch
from torch import nn
from torch import Tensor
from tqdm import trange
# Modules
from bnp4ood.openood_dataset_utils import setup_dataloaders, openood_eval, NUM_CLASSES


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
logger.addHandler(ch)

parser = argparse.ArgumentParser()
# Logging
parser.add_argument("--log_path", type=str, default="msp_odin.log")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--prefix", type=str, default="")
# Model
parser.add_argument("--autowhiten", action="store_true")
parser.add_argument("--autowhiten_factor", type=float, default=1e-7)
parser.add_argument("--autowhiten_dim", type=int, default=0)
parser.add_argument("--usepca", action="store_true")
parser.add_argument("--pca_dim", type=int, default=-1)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--T", type=float, default=1000.)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--wd", type=float, default=0.001)
parser.add_argument("--lr-min", type=float, default=0.0)
parser.add_argument("--lr-cosineannealing", action="store_true")
parser.add_argument("--features", type=str, default="vit-b-16")


def msp_score_fun(X: Float[Tensor, "num_samples dim"], net: nn.Module,
                  return_preds: bool = False, T=1.):
    """Helper to compute the MSP scores"""
    logits = net(X)
    probs = torch.softmax(logits / T, dim=1)
    scores, _ = torch.max(probs, axis=1)
    if return_preds:
        preds = torch.argmax(probs, axis=1)
        return scores, preds
    return scores


def main(args):
    # Logging
    fh = logging.FileHandler(args.log_path)
    logger.addHandler(fh)
    logger.info("Fit MSP on ViT Features for OpenOOD")
    logger.info("-----------------------------------")
    logger.info("Arguments:")
    logger.info(args)

    # Load datasets
    dataset_dict, id_feats, id_labels = setup_dataloaders(
        batch_size=args.batch_size,
        autowhiten=args.autowhiten,
        autowhiten_factor=args.autowhiten_factor,
        autowhiten_dim=args.autowhiten_dim,
        use_pca=args.usepca,
        pca_dim=args.pca_dim,
        features=args.features,
    )
    init_feats, _ = next(iter(dataset_dict["id"]["train"]))
    D = init_feats.shape[-1]

    # Train linear layer with CE
    torch.manual_seed(0)
    logger.info("Training linear layer with CE")
    net = nn.Linear(D, NUM_CLASSES, bias=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = None
    if args.lr_cosineannealing:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min
        )
    losses = []
    accs = []
    X = id_feats
    y = id_labels
    for _ in trange(args.epochs):
        optimizer.zero_grad()
        logits = net(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        acc = logits.argmax(1).eq(y).float().mean()
        losses.append(loss.item())
        accs.append(acc.item())

    net.eval()
    with torch.no_grad():
        train_acc = torch.argmax(net(X), axis=1).eq(y).float().mean().item()
        correct = 0.
        counts = 0.
        for X,y in dataset_dict["id"]["test"]:
            correct += torch.argmax(net(X), axis=1).eq(y).sum().item()
            counts += len(y)
        test_acc = correct / counts
    logger.info(f"Training Accuracy: {train_acc:.2%}")
    logger.info(f"Testing Accuracy: {test_acc:.2%}")

    # Evaluate
    T = args.T
    msp_output = dict(losses=losses, train_accuracies=accs, train_accuracy=train_acc,
                      test_accuracy=test_acc, state_dict=net.state_dict())
    temp_msp_output = dict(losses=losses, train_accuracies=accs, train_accuracy=train_acc, T=T,
                      test_accuracy=test_acc, state_dict=net.state_dict())
    logger.info("Evaluating on OpenOOD benchmark ")
    net = net.to(DEVICE)
    temp_msp_aurocs = openood_eval(dataset_dict, (net,), msp_score_fun, model=f"Temp. Scaled MSP {T}", T=T)
    msp_aurocs = openood_eval(dataset_dict, (net,), msp_score_fun, model=f"MSP")
    msp_output["openood"] = msp_aurocs
    temp_msp_output["openood"] = temp_msp_aurocs
    logger.info("MSP AUROC:")
    for k, v in msp_aurocs.items():
        logger.info(f"{k}: {v}")
    logger.info(f"Temperature Scaled MSP AUROC T={T}:")
    for k, v in temp_msp_aurocs.items():
        logger.info(f"{k}: {v}")
    torch.save(msp_output, os.path.join(args.output_path, args.prefix + "msp.result"))
    torch.save(temp_msp_output, os.path.join(args.output_path, args.prefix + "temp_msp.result"))


if __name__ == "__main__":
    main(parser.parse_args())

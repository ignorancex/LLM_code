"""Dataset utilities for loading and processing the OpenOOD datasets."""
# Python
import logging
from typing import Callable
# Third Party
from jaxtyping import Float, Int
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import torch
from torch import Tensor
from tqdm import tqdm


logger = logging.getLogger(__name__)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

# Store feature files
_template = "-img1k-{}-feats.pkl"
NUM_CLASSES = 1000
TRAIN_FNAME = "-img1k-feats.pkl"
VAL_FNAME = _template.format("val")
TEST_FNAME = _template.format("test")
OOD_NEAR_SSB_HARD = _template.format("near-ssb_hard")
OOD_NEAR_NINCO = _template.format("near-ninco")
OOD_FAR_INAT = _template.format("far-inaturalist")
OOD_FAR_OPENIMGO = _template.format("far-openimage_o")
OOD_FAR_TEXTURES = _template.format("far-textures")
DATASET_FEATFILES = dict(
    id=dict(
        train=TRAIN_FNAME,
        test=TEST_FNAME,
        val=VAL_FNAME,
    ),
    ood=dict(
        near=dict(
            ssb_hard=OOD_NEAR_SSB_HARD,
            ninco=OOD_NEAR_NINCO,
        ),
        far=dict(
            inaturalist=OOD_FAR_INAT,
            openimage_o=OOD_FAR_OPENIMGO,
            textures=OOD_FAR_TEXTURES,
        ),
    ),
)


def load_vit_feats(fname: str) -> tuple[
        Float[Tensor, "num_samples dim"], Int[Tensor, "num_samples"]]:
    dat = torch.load(fname)
    return dat["feats"], dat["labels"]


def get_dataloader(
        feats: Float[Tensor, "num_samples dim"],
        labels: Int[Tensor, "num_samples"],
        batch_size: int=1024,
        shuffle: bool=True,
    ) -> torch.utils.data.DataLoader:
    dset = torch.utils.data.TensorDataset(feats, labels)
    ldr = torch.utils.data.DataLoader(dset, batch_size, shuffle=shuffle)
    return ldr


def auto_whiten(
            feats: Float[Tensor, "num_samples dim"],
            labels: Int[Tensor, "num_samples"],
            autowhiten_factor: float = 1e-7,
            autowhiten_dim: int = 0,
    ) -> tuple[Float[Tensor, "num_samples newdim"], Callable]:
    """Marginal covariance whitening and dropping of degenerate dimensions
    followed by projection into the eigenspace of the average within-class
    covariance.
    """
    C = torch.cov(feats.T)
    evals, evecs = torch.linalg.eigh(C) # Sorted in ascending order
    evals = evals.flip(0)
    evecs = evecs.flip(1)
    thresh = evals.max() * C.shape[-1] * autowhiten_factor
    if autowhiten_dim > 0:
        keep = torch.zeros(evals.shape[0], dtype=torch.bool)
        keep[:autowhiten_dim] = True
    else:
        keep = evals > thresh

    logger.info(f"Auto Whitening (factor={autowhiten_factor}, dim={autowhiten_dim}): Keeping {keep.sum()} of {keep.shape[0]}")
    X_mean = feats.mean(0)
    whiten = lambda X: (X - X_mean) @ evecs[:, keep] / evals[keep].sqrt()
    X_whitened = whiten(feats)

    # Project into eigenbasis of average within-class covariance
    muk_hat = torch.stack([X_whitened[labels == k].mean(0) for k in range(NUM_CLASSES)])
    diff = X_whitened - muk_hat[labels]
    Psi0_star = torch.einsum("ni,nj->ij", diff, diff) / len(X_whitened)
    evalsP, evecsP = torch.linalg.eigh(Psi0_star)
    proj = lambda X: X @ evecsP
    Z = proj(X_whitened)
    return Z, lambda X: proj(whiten(X))


def setup_dataloaders(
        dataset_featfiles: dict = DATASET_FEATFILES,
        batch_size: int = 1024,
        shuffle : bool = True,
        autowhiten: bool = False,
        autowhiten_factor: float = 1e-7,
        autowhiten_dim: int = 0,
        use_pca: bool = False,
        pca_dim: int = -1,
        features: str = "vit-b-16",
    ) -> dict:
    def fname2loader(fname):
        feats, labels = load_vit_feats(fname)
        feats = preprocessor(feats)
        return get_dataloader(feats, labels, batch_size=batch_size, shuffle=shuffle)

    logger.info(f"Loading datasets with features: {features}")
    datasets = {"id": dict(), "ood": dict()}
    # Load train set first to setup PCA / whitening
    feats, labels = load_vit_feats(features + dataset_featfiles["id"]["train"])
    preprocessor = lambda X: X
    if autowhiten:
        feats, preprocessor = auto_whiten(feats, labels, autowhiten_factor, autowhiten_dim=autowhiten_dim)
    elif use_pca:
        logger.info(f"Using PCA with {pca_dim} components")
        pca = PCA(n_components=pca_dim)
        pca = pca.fit(feats)
        feats = pca.transform(feats)
        feats = torch.as_tensor(feats, dtype=torch.float32)
        preprocessor = lambda X: torch.as_tensor(pca.transform(X), dtype=torch.float32)
    ldr = get_dataloader(feats, labels, batch_size=batch_size, shuffle=shuffle)
    datasets["id"]["train"] = ldr

    for base_k, base_v in dataset_featfiles.items():
        for k, v in base_v.items():
            if isinstance(v, dict):
                datasets[base_k][k] = dict()
                for inner_k, inner_v in v.items():
                    datasets[base_k][k][inner_k] = fname2loader(features + inner_v)
            else:
                if k in ["train", "val"]:
                    continue
                datasets[base_k][k] = fname2loader(features + v)
    return datasets, feats, labels


def get_sufficient_stats(
        loader: torch.utils.data.DataLoader = None,
        X: Float[Tensor, "num_samples dim"] = None,
        Y: Int[Tensor, "num_samples"] = None,
        K: int = NUM_CLASSES,
    ) -> tuple[Float[Tensor, "K"], Float[Tensor, "K dim"], Float[Tensor, "K dim"]]:
    """Extract the sufficient statistics from a dataset"""
    logger.info("Extracting sufficient statistics")
    if loader is None and (X is None or Y is None):
        raise ValueError("Either loader or X and Y must be provided")
    if loader is not None:
        init_batch, _ = next(iter(loader))
        D = init_batch.shape[-1]
        with torch.no_grad():
            sumx = torch.zeros(K, D, device=DEVICE)
            sumxxT = torch.zeros(K, D, D, device=DEVICE)
            sumxsq = torch.zeros(K, D, device=DEVICE)
            Nk = torch.zeros(K, device=DEVICE)
            for x, y in tqdm(loader, leave=True, desc="Calculating sufficient stats"):
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                for k in range(K):
                    mask = y == k
                    sumx[k] += x[mask].sum(0)
                    sumxxT[k] += torch.einsum("ij,ik->jk", x[mask], x[mask])
                    sumxsq[k] += (x[mask] ** 2).sum(0)
                    Nk[k] += mask.sum()
    else:
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)
        Nk = torch.bincount(Y).float()
        sumx = torch.stack([X[Y == k].sum(0) for k in range(NUM_CLASSES)])
        sumxxT = torch.stack([torch.einsum("ij,ik->jk", X[Y == k], X[Y == k]) for k in range(NUM_CLASSES)])
        sumxsq = torch.stack([(X[Y == k]**2).sum(0) for k in range(NUM_CLASSES)])
    return Nk, sumx, sumxxT, sumxsq


def calc_scores(loader: torch.utils.data.DataLoader,
                params: tuple,
                score_fn: Callable,
                return_accuracy: bool=False,
                model: str="",
                dset: str="",
                **kwargs
               ) -> Float[Tensor, "N"] | tuple[Float[Tensor, "N"], float]:
    all_scores = torch.empty(0)
    correct = 0.
    count = 0.
    with torch.no_grad():
        for x, y  in tqdm(loader, leave=True, desc=f"Calculating {model} scores {dset}"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            scores = score_fn(x, *params, return_preds=return_accuracy, **kwargs)
            if return_accuracy:
                scores, preds = scores
                correct += preds.eq(y).sum()
                count += x.shape[0]
            all_scores = torch.cat((all_scores, scores.cpu()), 0)
    if return_accuracy:
        return all_scores, correct / count
    return all_scores


def openood_eval(dataset_dict, params,
                 score_fn: Callable, model="", **kwargs) -> dict:
    id_scores, acc = calc_scores(dataset_dict["id"]["test"],
                                 params,
                                 score_fn,
                                 return_accuracy=True,
                                 dset="ID", model=model, **kwargs)
    id_scores = id_scores.numpy()
    aurocs = {"acc": acc}
    for difficulty, dsets in dataset_dict["ood"].items():
        logger.info(f"Calculating {difficulty} OOD scores")
        n = 0.
        sumauc = 0.
        for dset, loader in dsets.items():
            ood_scores = calc_scores(loader, params, score_fn, dset=dset, model=model, **kwargs)
            ood_scores = ood_scores.numpy()
            all_scores = np.concatenate((id_scores, ood_scores), 0)
            isid = np.zeros(all_scores.shape[0])
            isid[:id_scores.shape[0]] = 1
            auc = roc_auc_score(isid, all_scores)
            aurocs[dset] = auc
            sumauc += auc
            n += 1
        aurocs[difficulty] = sumauc / n
    return aurocs

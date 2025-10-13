import os
import random
import tqdm
import wandb
import torch
import subprocess
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from functools import partial
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Callable, List
from sklearn import metrics
from sklearn.model_selection import train_test_split

from source.dataset_utils import collate_features

from query_strategies.random_sampling import RandomSampling
from query_strategies.entropy_sampling import EntropySampling
from query_strategies.badge_sampling import BadgeSampling
from query_strategies.core_set import Coreset
from query_strategies.kmeans_sampling import KMeansSampling
from query_strategies.cdal_sampling import CDALSampling

strategy_dict = {
    "RandomSampling": RandomSampling,
    "EntropySampling": EntropySampling,
    "BadgeSampling": BadgeSampling,
    "Coreset": Coreset,
    "KMeansSampling": KMeansSampling,
    "CDALSampling": CDALSampling,
}

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def write_dictconfig(d, f, child: bool = False, ntab=0):
    for k, v in d.items():
        if isinstance(v, dict):
            if not child:
                f.write(f"{k}:\n")
            else:
                for _ in range(ntab):
                    f.write("\t")
                f.write(f"- {k}:\n")
            write_dictconfig(v, f, True, ntab=ntab + 1)
        else:
            if isinstance(v, list):
                if not child:
                    f.write(f"{k}:\n")
                    for e in v:
                        f.write(f"\t- {e}\n")
                else:
                    for _ in range(ntab):
                        f.write("\t")
                    f.write(f"{k}:\n")
                    for e in v:
                        for _ in range(ntab):
                            f.write("\t")
                        f.write(f"\t- {e}\n")
            else:
                if not child:
                    f.write(f"{k}: {v}\n")
                else:
                    for _ in range(ntab):
                        f.write("\t")
                    f.write(f"- {k}: {v}\n")


def initialize_wandb(
    cfg: DictConfig,
    tags: Optional[List] = None,
    key: Optional[str] = "",
):
    command = f"wandb login {key}"
    subprocess.call(command, shell=True)
    if tags == None:
        tags = []
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.username,
        name=cfg.wandb.exp_name,
        group=cfg.wandb.group,
        dir=cfg.wandb.dir,
        config=config,
        tags=tags,
    )
    config_file_path = Path(run.dir, "run_config.yaml")
    d = OmegaConf.to_container(cfg, resolve=True)
    with open(config_file_path, "w+") as f:
        write_dictconfig(d, f)
        wandb.save(str(config_file_path))
        f.close()
    return run

def initialize_df(slide_ids):
    nslide = len(slide_ids)
    df_dict = {
        "slide_id": slide_ids,
        "process": np.full((nslide), 1, dtype=np.uint8),
        "status": np.full((nslide), "tbp"),
    }
    df = pd.DataFrame(df_dict)
    return df


def extract_coord_from_path(path):
    """
    Path expected to look like /path/to/dir/x_y.png
    """
    x_y = path.stem
    x, y = x_y.split("_")[0], x_y.split("_")[1]
    return int(x), int(y)


def update_state_dict(model_dict, state_dict):
    success, failure = 0, 0
    updated_state_dict = {}
    for k, v in zip(model_dict.keys(), state_dict.values()):
        if v.size() != model_dict[k].size():
            updated_state_dict[k] = model_dict[k]
            failure += 1
        else:
            updated_state_dict[k] = v
            success += 1
    msg = f"{success} weight(s) loaded succesfully ; {failure} weight(s) not loaded because of mismatching shapes"
    return updated_state_dict, msg


def create_train_tune_test_df(
    df: pd.DataFrame,
    save_csv: bool = False,
    output_dir: Path = Path(""),
    tune_size: float = 0.4,
    test_size: float = 0.2,
    seed: Optional[int] = 21,
):
    train_df, tune_df = train_test_split(
        df, test_size=tune_size, random_state=seed, stratify=df.label
    )
    test_df = pd.DataFrame()
    if test_size > 0:
        train_df, test_df = train_test_split(
            train_df, test_size=test_size, random_state=seed, stratify=train_df.label
        )
        test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    tune_df = tune_df.reset_index(drop=True)
    if save_csv:
        train_df.to_csv(Path(output_dir, f"train.csv"), index=False)
        tune_df.to_csv(Path(output_dir, f"tune.csv"), index=False)
        if test_size > 0:
            test_df.to_csv(Path(output_dir, f"test.csv"), index=False)
    return train_df, tune_df, test_df

def compute_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def calculate_stats(data: List[float]):
    mean = round(np.mean(data), 3)
    std = round(np.std(data), 3)
    return mean, std

def get_binary_metrics(probs: np.array(float), preds: List[int], labels: List[int]):
    labels = np.asarray(labels)
    acc = metrics.accuracy_score(labels, preds)
    auc = metrics.roc_auc_score(labels, probs)
    precision = metrics.precision_score(labels, preds, zero_division=0)
    recall = metrics.recall_score(labels, preds)
    metrics_dict = {
        "accuracy": acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
    }
    return metrics_dict

def get_metrics(
    probs: np.array(float),
    preds: List[int],
    labels: List[int],
    multi_class: str = "ovr",
):
    labels = np.asarray(labels)
    auc = metrics.roc_auc_score(labels, probs, multi_class=multi_class)
    quadratic_weighted_kappa = metrics.cohen_kappa_score(
        labels, preds, weights="quadratic"
    )
    metrics_dict = {"auc": auc, "kappa": quadratic_weighted_kappa}
    return metrics_dict

def collate_region_filepaths(batch):
    item = batch[0]
    idx = torch.LongTensor([item[0]])
    fp = item[1]
    return [idx, fp]


def get_roc_auc_curve(probs: np.array(float), labels: List[int], result_dir: Path):
    fpr, tpr, _ = metrics.roc_curve(labels, np.array(probs))
    auc = metrics.roc_auc_score(labels, probs)
    fig = plt.figure(dpi=600)
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Receiver Operating Characteristic (ROC) curve")
    plt.legend(loc="lower right")
    save_location = Path(result_dir, "roc_auc_curve.png")
    plt.savefig(save_location)
    plt.close()
    return save_location

def log_on_step(
    name, results, step: str = "epoch", to_log: Optional[List["str"]] = None
):
    if not to_log:
        to_log = list(results.keys())
    for r, v in results.items():
        if r in to_log:
            wandb.define_metric(f"{name}/{r}", step_metric=step)
            wandb.log({f"{name}/{r}": v})

def make_weights_for_balanced_classes(dataset):
    n_samples = len(dataset)
    weight_per_class = []
    for c in range(dataset.num_classes):
        w = n_samples * 1.0 / len(dataset.class_2_id[c])
        weight_per_class.append(w)
    weight = []
    for idx in range(len(dataset)):
        y = dataset.get_label(idx)
        weight.append(weight_per_class[y])
    return torch.DoubleTensor(weight)


def logit_to_ordinal_prediction(logits):
    with torch.no_grad():
        pred = torch.sigmoid(logits)
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1

class OptimizerFactory:
    def __init__(
        self,
        name: str,
        params: nn.Module,
        lr: float,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        nesterov: bool = True,
    ):

        if name == "adam":
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif name == "sgd":
            self.optimizer = optim.SGD(
                params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
            )
        else:
            raise KeyError(f"{name} not supported")

    def get_optimizer(self):
        return self.optimizer

class WarmupCosineLrScheduler(torch.optim.lr_scheduler._LRScheduler):
    '''
    This is different from official definition, this is implemented according to
    the paper of fix-match
    '''
    def __init__(
            self,
            optimizer,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = self.max_iter - self.warmup_iter
            ratio = np.cos((7 * np.pi * real_iter) / (16 * real_max_iter))
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio
    
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

class SchedulerFactory:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        params: Optional[dict] = None,
    ):

        self.scheduler = None
        self.name = params.name
        if self.name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=params.step_size, gamma=params.gamma
            )
        elif self.name == "cosine":
            assert (
                params.T_max != -1
            ), "T_max parameter must be specified! If you dont know what to use, plug in nepochs"
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                params.T_max, eta_min=params.eta_min
            )
        elif self.name == "reduce_lr_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=params.mode,
                factor=params.factor,
                patience=params.patience,
                min_lr=params.min_lr,
            )
        elif self.name:
            raise KeyError(f"{self.name} not supported")

    def get_scheduler(self):
        return self.scheduler


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        tracking: str,
        min_max: str,
        patience: int = 20,
        min_epoch: int = 50,
        checkpoint_dir: Optional[Path] = None,
        save_all: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement
        """
        self.tracking = tracking
        self.min_max = min_max
        self.patience = patience
        self.min_epoch = min_epoch
        self.checkpoint_dir = checkpoint_dir
        self.save_all = save_all
        self.verbose = verbose

        self.best_score = None
        self.early_stop = False

    def __call__(self, epoch, model, results):

        score = results[self.tracking]
        if self.min_max == "min":
            score = -1 * score

        if self.best_score is None or score >= self.best_score:
            self.best_score = score
            # fname = f"best_model_{wandb.run.id}.pt"
            fname = "best_model.pt"
            torch.save(model.state_dict(), Path(self.checkpoint_dir, fname))
            self.counter = 0

        elif score < self.best_score:
            self.counter += 1
            if epoch <= self.min_epoch + 1 and self.verbose:
                print(
                    f"EarlyStopping counter: {min(self.counter,self.patience)}/{self.patience}"
                )
            elif self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience and epoch > self.min_epoch:
                self.early_stop = True

        if self.save_all:
            fname = f"epoch_{epoch}.pt"
            torch.save(model.state_dict(), Path(self.checkpoint_dir, fname))

def next_candidates(
    model: nn.Module,
    pool_dataset: torch.utils.data.Dataset,
    train_dataset: torch.utils.data.Dataset,
    collate_fn: Callable = partial(collate_features, label_type="int"),
    batch_size: Optional[int] = 1,
    strategy_name: str = None,
    WSI_budget: int = 1,
):
    cls = strategy_dict[strategy_name]
    strategy = cls(model, pool_dataset, train_dataset, collate_fn, batch_size)

    selected = strategy.query(WSI_budget)
    
    slide_id_selected = [pool_dataset.get_slide_id(i) for i in selected]
    labels_selected = [pool_dataset.get_label(i) for i in selected]
    
    df = pd.DataFrame(list(zip(slide_id_selected, labels_selected)), columns=['slide_id', 'label'])
    candidates = [(slide_id, label) for slide_id, label in zip(slide_id_selected, labels_selected)]
    
    return df, candidates

def train(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    collate_fn=partial(collate_features, label_type="int"),
    batch_size: Optional[int] = 1,
    weighted_sampling: Optional[bool] = False,
    gradient_clipping: Optional[int] = None,
    mixing: Optional[DictConfig] = None,
    result_dir: Path = None,
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    epoch_loss = 0
    probs = np.empty((0, dataset.num_classes))
    pred_labels_list, labels_list = [], []
    idxs = []

    bce_loss = nn.BCELoss().cuda()
    softmax = nn.Softmax(dim=1).cuda()

    if weighted_sampling:
        weights = make_weights_for_balanced_classes(dataset)
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
    else:
        sampler = None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=not weighted_sampling,  # shuffle if not using a WeightedRandomSampler
        collate_fn=collate_fn,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Train - Epoch {epoch}"),
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=True,
    ) as t:

        for i, batch in enumerate(t):

            optimizer.zero_grad()

            idx, slide_id, features, mask, labels = batch

            features, mask, labels = features.to(device, non_blocking=True), mask.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if mixing is not None:
                embeddings, reweighted_target, logits = model(x=features, mask=mask, target=labels, mixup=mixing.mixup, manifold_mixup=mixing.manifold_mixup, mixup_type=mixing.mixup_type)
                loss = bce_loss(softmax(logits), reweighted_target)
            else:
                embeddings, zero_embeddings, logits = model(x=features, mask=mask)
                loss = criterion(logits, labels)
        
            epoch_loss += loss.item()

            if gradient_clipping:
                loss = loss / gradient_clipping

            loss.backward()
            optimizer.step()

            pred = torch.topk(logits, 1, dim=1)[1]
            pred_labels_list.extend(pred[:, 0].clone().tolist())

            prob = F.softmax(logits, dim=1).cpu().detach().numpy()
            probs = np.append(probs, prob, axis=0)

            labels_list.extend(labels.clone().tolist())

            idxs.extend(list(idx))

    # TODO: what happens if idxs is not made of unique index values?
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()

    if dataset.num_classes == 2:
        metrics = get_binary_metrics(probs[:, 1], pred_labels_list, labels_list)
        roc_auc_curve_path = get_roc_auc_curve(probs[:, 1], labels_list, result_dir)
        results.update({"roc_auc_curve_path": roc_auc_curve_path})
    else:
        metrics = get_metrics(probs, pred_labels_list, labels_list)

    results.update(metrics)

    train_loss = epoch_loss / len(loader)
    results["loss"] = train_loss

    return results

def test(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    collate_fn=partial(collate_features, label_type="int"),
    batch_size: Optional[int] = 1,
    result_dir: Path = None,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    probs = np.empty((0, dataset.num_classes))
    pred_labels_list, labels_list = [], []
    idxs = []

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Test"),
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=True,
    ) as t:

        with torch.no_grad():

            for i, batch in enumerate(t):

                idx, slide_id, features, mask, labels = batch

                features, mask, labels = features.to(device, non_blocking=True), mask.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                embeddings, zero_embeddings, logits = model(x=features, mask=mask)

                pred = torch.topk(logits, 1, dim=1)[1]
                pred_labels_list.extend(pred[:, 0].clone().tolist())

                prob = F.softmax(logits, dim=1).cpu().detach().numpy()
                probs = np.append(probs, prob, axis=0)

                labels_list.extend(labels.clone().tolist())
                idxs.extend(list(idx))

    # TODO: what happens if idxs is not made of unique index values?
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()

    if dataset.num_classes == 2:
        metrics = get_binary_metrics(probs[:, 1], pred_labels_list, labels_list)
        roc_auc_curve_path = get_roc_auc_curve(probs[:, 1], labels_list, result_dir)
        results.update({"roc_auc_curve_path": roc_auc_curve_path})
    else:
        metrics = get_metrics(probs, pred_labels_list, labels_list)

    results.update(metrics)

    return results

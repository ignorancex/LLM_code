import logging
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.utilities import rank_zero_only

import matplotlib.pyplot as plt
# import scienceplots
# plt.style.use(['science', 'tableau-colorblind10']) # causing some issues with 'cmr10.tfm' file while using pytorch module from setonix

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# collected from https://github.com/martius-lab/beta-nll
def beta_nll_loss(mean, variance, target, beta=0.5):
    """Compute beta-NLL loss
    
    :param mean: Predicted mean of shape B x D or B
    :param variance: Predicted variance of shape B x D or B
    :param target: Target of shape B x D or B
    :param beta: Parameter from range [0, 1] controlling relative 
        weighting between data points, where `0` corresponds to 
        high weight on low error points and `1` to an equal weighting.
    :returns: Loss per batch element of shape B
    """
    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

    if beta > 0:
        loss = loss * (variance.detach() ** beta)
    
    return loss.sum(axis=-1)

# collected and slightly updated from https://github.com/Lightning-AI/pytorch-lightning/issues/16881#issuecomment-1447429542
class DelayedStartEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set start_epoch to None or 0 for no delay
        self.start_epoch = start_epoch

    def on_train_epoch_end(self, trainer: L.Trainer, l_module: L.LightningModule) -> None:
        if (self.start_epoch is not None) and (trainer.current_epoch < self.start_epoch):
            return
        super().on_train_epoch_end(trainer, l_module)

    def on_validation_end(self, trainer: L.Trainer, l_module: L.LightningModule) -> None:
        if (self.start_epoch is not None) and (trainer.current_epoch < self.start_epoch):
            return
        super().on_validation_end(trainer, l_module)

def plot_uncertainy(data: dict, save_as: str) -> None:
    mean = data.get("mean", 0)
    unc = np.sqrt(data.get("var", 0))
    labels = data.get("labels", 0)
    # noise = data.get("noise", 0)

    _, ax = plt.subplots(1, 1)
    sc = ax.scatter(labels, mean, c=unc, cmap="coolwarm")
    # ax.errorbar(labels, mean, yerr=unc, fmt='o', alpha=0.5)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    
    cbar = plt.colorbar(sc)
    cbar.set_label("Uncertainty")

    plt.savefig(save_as)
    plt.close()
    log_info(logger, f"Saved plot at {save_as}")

def read_newsemp_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".tsv"):
        df = pd.read_csv(file_path, sep='\t', na_values="unknown") # some column includes "unknown"
    elif file_path.endswith(".csv"):
        # 2024 raw data are in csv. The essay has commas, and placed inside double quotes
        # Further, tt has \" inside the quoted text, for example, "I am a \"good\" person"
        df = pd.read_csv(file_path, quotechar='"', escapechar="\\")

        # "2024" has different column names
        df = df.rename(columns={
            "person_essay": "essay",
            "person_empathy": "empathy"
        })
    else:
        raise ValueError(f"File extension not supported: {file_path}")
    return df

def process_seedwise_metrics(results: list, save_as: str) -> None:
    results_df = pd.DataFrame(results)

    results_df.set_index("seed", inplace=True)
    results_df = results_df.round(3)
    
    # post-processing
    mean_row = results_df.mean(numeric_only=True).round(3)
    std_row = results_df.std(numeric_only=True).round(3)
    median_row = results_df.median(numeric_only=True).round(3)
    
    best_scores = pd.DataFrame(index=["best"], columns=results_df.columns)
    for col in results_df.columns:
        if col.endswith("_rmse"):
            best_scores.loc["best", col] = results_df[col].min()
        else:
            best_scores.loc["best", col] = results_df[col].max()
    best_row = best_scores.loc["best", :].round(3)
    
    # Assign a label to identify each row
    mean_row.name = "mean"
    std_row.name = "std"
    median_row.name = "median"

    results_df = pd.concat([results_df, mean_row.to_frame().T, std_row.to_frame().T, median_row.to_frame().T])
    
    results_df.to_csv(save_as, index=True)
    log_info(logger, f"Results saved at {save_as}")

    selected_cols = [col for col in results_df.columns if col.startswith("test")]

    if len(selected_cols) == 0:
        # if no test results, use val results
        selected_cols = [col for col in results_df.columns if col.startswith("val")]

    # print the result, in LaTeX-table style
    log_info(logger, f"Median(Best) {' & '.join(selected_cols)}")
    log_info(logger, " & ".join([f"${median}({best})$" \
                                 for median, best in zip(median_row[selected_cols], best_row[selected_cols])]))

@rank_zero_only
def log_info(logger, msg):
    logger.info(msg)

@rank_zero_only
def log_debug(logger, msg):
    logger.debug(msg)

def retrieve_newsemp_file_names(config: OmegaConf) -> tuple[list, list, list]:
    llm = "llama"
    train_attr = f"train_{llm}"
    val_attr = f"val_{llm}"

    if config.expt.val_data == 2024:
        test_attr = f"test_{llm}"
    elif config.expt.val_data == 2022:
        test_attr = f"test"
    else:
        raise ValueError(f"Validation {config.expt.val_data} is not configured for test.")
    
    train_file_list = []
    for data in config.expt.train_data:
        train_file_list.append(getattr(config[data], train_attr))
        if data != config.expt.val_data:
            # we don't want to include val data in the training data for the same year
            train_file_list.append(getattr(config[data], val_attr))

    val_file_list = [getattr(config[config.expt.val_data], val_attr)]
    test_file_list = [getattr(config[config.expt.val_data], test_attr)]

    return train_file_list, val_file_list, test_file_list

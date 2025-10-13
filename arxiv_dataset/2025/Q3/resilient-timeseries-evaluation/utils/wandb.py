import wandb
from pytorch_lightning.loggers import WandbLogger
import datetime

def init_wandb(wandb_project_name : str, model_type : str, model_output_dir : str, wandb_offline : bool) -> WandbLogger:
    """
    Initialize WandbLogger for logging experiment metrics and parameters.

    Parameters
    ----------
    args : object
        An object containing the arguments for the experiment.
    model_output_path : str
        The directory where the model output will be saved.
    wandb_online : bool
        Whether wandb should log online.

    Returns
    -------
    WandbLogger
        The initialized WandbLogger object.

    """
    # Finish a previous run if that hasn't been correctly finished
    if wandb.run is not None:
        wandb.finish()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_logger = WandbLogger(project=wandb_project_name, name=f"{model_type}_{timestamp}", save_dir=model_output_dir, offline=wandb_offline)
    wandb.init(project=wandb_project_name, name=f"{model_type}_{timestamp}", mode = 'offline' if wandb_offline else 'online')
        
    return wandb_logger

def finish_wandb() -> None:
    """
    The `finish_wandb` function checks if there is an active Weights & Biases run and finishes it if so.
    """
    if wandb.run is not None:
        wandb.finish()
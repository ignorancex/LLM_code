import os
from datetime import datetime
import torch
import time

def create_result_dir(global_res_path, middle_args:list=None):
    
    if not os.path.exists(global_res_path):
        os.makedirs(global_res_path)
        print(f"Created directory: {global_res_path}")
    else:
        print(f"Directory already exists: {global_res_path}")
    
    dir_made = False
    while not dir_made:
        try:
            # Create a directory for the current run
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            if middle_args is not None:
                assert os.path.exists(os.path.join(global_res_path, *middle_args)), f"Directory {os.path.join(global_res_path, *middle_args)} does not exist!"
                current_run_path = os.path.join(global_res_path, *middle_args, current_time)
            else:
                current_run_path = os.path.join(global_res_path, current_time)
            
            os.makedirs(current_run_path)

            dir_made = True
            break
        except FileExistsError:
            print("Directory already exists, retrying...")
            time.sleep(1)

    print(f"Created directory: {current_run_path}")

    return current_run_path

def get_device(use_accelerator):
    device = torch.device("cpu")
    if use_accelerator:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("No CUDA or MPS available, using CPU instead.")
    
    return device

def save_chkpt(model, optimizer, scheduler, loss_hist, best_loss, no_improvement, chkpt_path, task_specific_metric_hist=None):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_hist': loss_hist,
        'best_loss': best_loss,
        'no_improvement': no_improvement,
        'task_specific_metric_hist': task_specific_metric_hist
    }, chkpt_path)
    print(f"Checkpoint saved at {chkpt_path}")
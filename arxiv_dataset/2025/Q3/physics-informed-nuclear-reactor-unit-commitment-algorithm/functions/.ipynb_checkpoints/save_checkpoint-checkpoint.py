import os
import pandas as pd

def save_checkpoint(n, start_day, checkpoint_dir,
                    master_df, GEN_df, SHUT_df, START_df,
                    master_shut_df, master_commit, master_kinfTable):
    """
    Save intermediate results for rolling UC runs into checkpoint directory.
    Deletes the previous checkpoint to save space.
    """
    results_path = os.path.join(checkpoint_dir, f'checkpoint_{n}')
    os.makedirs(results_path, exist_ok=True)

    # Save all results
    master_df.to_csv(os.path.join(results_path, 'aggregate_results.csv'))
    GEN_df.to_csv(os.path.join(results_path, 'GEN_results.csv'))
    SHUT_df.to_csv(os.path.join(results_path, 'SHUT_results.csv'))
    START_df.to_csv(os.path.join(results_path, 'START_results.csv'))
    master_shut_df.to_csv(os.path.join(results_path, 'shut_count_and_obj_value.csv'))
    master_commit.to_csv(os.path.join(results_path, 'commit_results.csv'))
    master_kinfTable.to_csv(os.path.join(results_path, 'master_kinfTable.csv'))

    # Delete previous checkpoint (to save storage)
    if n > start_day:
        prev_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{n-1}')
        if os.path.exists(prev_checkpoint_path):
            for file in os.listdir(prev_checkpoint_path):
                os.remove(os.path.join(prev_checkpoint_path, file))
            os.rmdir(prev_checkpoint_path)

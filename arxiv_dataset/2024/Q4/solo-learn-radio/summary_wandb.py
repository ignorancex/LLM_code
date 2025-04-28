import wandb
import pandas as pd
import csv

# Autenticazione a WandB (se necessario)
wandb.login()

# Specifica il tuo username e il nome del progetto
username = "thomas-cecconello"
project_name = "solo-learn"

# Recupera le run
runs = wandb.Api().runs(f"{username}/{project_name}")

# Crea un dizionario per memorizzare le run più recenti per ogni nome
latest_runs = {}

# Itera su tutte le run
for run in runs:
    #print(run.name)
    # Controlla se il nome della run corrisponde al pattern e se la run è terminata
    if run.state == "finished" and "__" in run.name:
        # Se la run non è nel dizionario o se è più recente di quella memorizzata
        if run.name not in latest_runs or run.createdAt > latest_runs[run.name].createdAt:
            # Memorizza la run
            latest_runs[run.name] = run
            

# Crea una lista per memorizzare le accuracy
run_details = []

# Itera attraverso le run e recupera le informazioni desiderate
for run in latest_runs.values():
    if run.state == "finished" and "__" in run.name:
        info = run.name.split("__")
        #table = run.summary['mean_std_table']
        table = wandb.Api().artifact(f"{username}/{project_name}/run-{run.id}-mean_std_table:latest").get("mean_std_table")
        
        if table is not None:
            mean = table.get_column("mean")[0]
            std = table.get_column("std")[0]
            print(mean)
            
            run_info = {
                'eval_data': info[2],
                'eval_aug': info[3],
                'accuracy_mean': mean,
                'accuracy_std': std,
                'eval_method': info[1],
                'pretrain_method': info[4],
                'K_fold': info[5],
                'balancing_strategy': info[6],
                'datalist': info[7],
                "run_id": run.id,
                "run_name": run.name
            }
            run_details.append(run_info)
            print(run_info)

column_names = run_details[0].keys()

with open('exp_summary.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=column_names)
    writer.writeheader()
    writer.writerows(run_details)
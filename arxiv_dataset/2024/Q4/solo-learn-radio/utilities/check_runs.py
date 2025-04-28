import wandb
import pandas as pd
import csv
import json

# Autenticazione a WandB (se necessario)
wandb.login()

# Specifica il tuo username e il nome del progetto
username = "thomas-cecconello"
project_name = "solo-learn"

# Recupera le run
runs = wandb.Api().runs(f"{username}/{project_name}")

# Crea un dizionario per memorizzare le run più recenti per ogni nome
latest_runs = {}

"""
check superhulk
for run in runs:
    #print(run.name)
    # Controlla se il nome della run corrisponde al pattern e se la run è terminata
    if "superhulk" in run.name:
        print(run.name)
        config = json.loads(run.json_config)
        accuracies_table = wandb.Api().artifact(f"{username}/{project_name}/run-{run.id}-accuracies_table:latest").get("accuracies_table")
        accuracies = accuracies_table.get_column("accuracy_values")
        print(accuracies)
"""

"""
check pretrains
for run in runs:
    #print(run.name)
    # Controlla se il nome della run corrisponde al pattern e se la run è terminata
    config = json.loads(run.json_config)
    #print(run.name)
    try:
        if config['data']['value']['dataset'] in ["banner", "hulk"]:
            if "wmse" in run.name:
                print(run.name)
    except:
        print("no data key")
"""




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
        try:
            config = json.loads(run.json_config)
            info = run.name.split("__")
            pretrain_info = config['pretrained_feature_extractor']['value'].split("/")
            pretrain_run = wandb.Api().run(f"{username}/{project_name}/{pretrain_info[2]}")
            pretrain_config = json.loads(pretrain_run.json_config)
            #table = run.summary['mean_std_table']
            try:
                #mean_std_table = wandb.Api().artifact(f"{username}/{project_name}/run-{run.id}-mean_std_table:latest").get("mean_std_table")
                accuracies_table = wandb.Api().artifact(f"{username}/{project_name}/run-{run.id}-accuracies_table:latest").get("accuracies_table")
            
                if accuracies_table is not None:
                    #mean = table.get_column("mean")[0]
                    #std = table.get_column("std")[0]
                    accuracies = accuracies_table.get_column("accuracy_values")
                    print(accuracies)
                    
                    run_info = {
                        'eval_id': run.id,
                        'eval_name': run.name,
                        'eval_dataset': config['data']['value']['dataset'],
                        'eval_aug': 'minmax' if config["augmentations"]["value"][0]["norm_channels"]["ch0"] == ['minmax'] else "mixed",
                        'eval_method': info[1],
                        'eval_K_fold': 'finetune' if config['finetune']['value'] else 'linear',
                        'pretrain_id': pretrain_run.id,
                        'pretrain_name': pretrain_run.name,
                        'pretrain_dataset': pretrain_config['data']['value']['dataset'],
                        'pretrain_method': pretrain_config['method']['value'],
                        'pretrain_aug': 'minmax' if pretrain_config["augmentations"]["value"][0]["norm_channels"]["ch0"] == ['minmax'] else "mixed",
                        'accuracies': accuracies,
                        'backbone': config['backbone']["value"]["name"],
                        'K': config['K_fold']["value"]
                    }
                    run_details.append(run_info)
                    print(run_info)
            except:
                print(f"No accuracy table for {run.name}")
        except:
            print(f"Boh {run.name}")

column_names = run_details[0].keys()

with open('exp_summary.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=column_names)
    writer.writeheader()
    writer.writerows(run_details)
import pandas as pd
import itertools
import wandb
import csv
import json

# Define the possible values for each factor
IDs = ["_"]
methods = ['byol', 'simclr', 'swav', 'dino', 'wmse', 'all4one']
backbones = ['resnet18', 'resnet50']
augmentations = ['minmax', 'mixed']
datasets = ['banner', 'hulk']

val_datasets = ['robin', 'vlass', 'rgz', 'mirabest', 'frg']
val_augmentations = ['minmax', 'mixed']
val_types = ['linear', 'finetune']

# Generate all possible combinations of the factors
all_combinations = list(itertools.product(IDs, methods, datasets, backbones, augmentations))

# Create a DataFrame from these combinations
df = pd.DataFrame(all_combinations, columns=['ID','Method', 'Dataset', 'Backbone', 'Augmentation'])

# Add a column to indicate if the experiment has been run (initially set to False)
df['Done'] = False

# Generate validation columns
validation_combinations = list(itertools.product(val_types, val_datasets, val_augmentations))
validation_columns = [f'{val_type}_{val_dataset}_{val_aug}' for val_type, val_dataset, val_aug in validation_combinations]

for col in validation_columns:
    df[col] = False

# Display the DataFrame
print(df)


# Define a function to update the experiment status
def mark_experiment_run(df, method, dataset, backbone, augmentation):
    df.loc[
        (df['Method'] == method) &
        (df['Backbone'] == backbone) &
        (df['Augmentation'] == augmentation) &
        (df['Dataset'] == dataset), 'Done'
    ] = True

def add_id(df, run_id, method, dataset, backbone, augmentation):
    df.loc[
        (df['Method'] == method) &
        (df['Backbone'] == backbone) &
        (df['Augmentation'] == augmentation) &
        (df['Dataset'] == dataset), 'ID'
    ] = run_id

# Define a function to update the experiment status
def mark_eval_run(df, eval_dataset, eval_augmentation, eval_type, method, dataset, backbone, augmentation):
    df.loc[
        (df['Method'] == method) &
        (df['Backbone'] == backbone) &
        (df['Augmentation'] == augmentation) &
        (df['Dataset'] == dataset), f'{eval_type}_{eval_dataset}_{eval_augmentation}'
    ] = True
    #print(f'{eval_type}_{eval_dataset}_{eval_augmentation}')

# Autenticazione a WandB (se necessario)
wandb.login()

# Specifica il tuo username e il nome del progetto
username = "ssl-inaf"

project_name = "solo-learn"

# Recupera le run
runs = wandb.Api().runs(f"{username}/{project_name}")

pretraining_runs = []
other_runs = []

for run in runs:
    config = json.loads(run.json_config)
    # Check if the 'method' key exists in the configuration
    try: 
        if config['method']:
            pretraining_runs.append(run)            
    except:
        other_runs.append(run)

print(f"{len(pretraining_runs)}")
print(f"{len(other_runs)}")

for idx, run in enumerate(pretraining_runs):
    config = json.loads(run.json_config)
    # If method key exists, then is a pretraining experiment
    try:
        if config['method']:
            #print(f'Pretrain {idx}')
            # Example: Mark the experiment as run
            run_id          = run.id
            dataset         = config['data']['value']['dataset']
            method          = config['method']['value']
            augmentation    = 'minmax' if config["augmentations"]["value"][0]["norm_channels"]["ch0"] == ['minmax'] else "mixed"
            backbone        = config['backbone']["value"]["name"]
            mark_experiment_run(df, method, dataset, backbone, augmentation)
            add_id(df, run_id, method, dataset, backbone, augmentation)
            for eval_idx, eval_run in enumerate(other_runs):
                #print(f'Eval {eval_idx}')
                eval_config = json.loads(eval_run.json_config)
                if 'pretrained_feature_extractor' in eval_config:
                    if run_id in eval_config['pretrained_feature_extractor']['value']:
                        print(f'FOUND')
                        #print(f'{eval_config}')
                        eval_dataset        = eval_config['data']['value']['dataset']
                        eval_augmentation   = 'minmax' if eval_config["augmentations"]["value"][0]["norm_channels"]["ch0"] == ['minmax'] else "mixed"
                        eval_type           = 'finetune' if eval_config["finetune"]["value"] == True else 'linear'
                        mark_eval_run(df, eval_dataset, eval_augmentation, eval_type, method, dataset, backbone, augmentation)
    except Exception as e:
        pass

# Display the updated DataFrame

df= df.drop(df[(df['Backbone'] == "resnet50") & (df['Dataset'] == "hulk")].index)
print(df)
# Save the DataFrame to a CSV file
df.to_csv('experiments_status.csv', index=False)

# Function to apply conditional formatting
def color_experiment_run(val):
    color = 'green' if val else 'red'
    return f'background-color: {color}'

# Apply the conditional formatting
#styled_df = df.style.applymap(color_experiment_run, subset=[])

print(validation_columns)

print(validation_columns+['Done'])
styled_df = df.style.applymap(color_experiment_run, subset=validation_columns+['Done'])

# Save the styled DataFrame to an HTML file
styled_df.to_html('experiments_status.html', index=False)

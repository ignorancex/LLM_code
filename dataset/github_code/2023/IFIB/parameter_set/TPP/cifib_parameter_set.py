# parameter sets of model ifib

citibike_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "generic_continuous", \
    "--dataloader_config", "citibike/cifib_dl.json",
    "--dataset_name", "citibike", \
    "--n_training_steps", "100000", \
    "--n_evaluation_steps", "2000", \
    "--n_report_steps", "2000", \
    "--b", "32", \
    "--n_warmup_steps", "20000", \
    "--model_name", "cifib", \
    "--model_config", "citibike/cifib.json",
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",\
]

covid19_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "generic_continuous", \
    "--dataloader_config", "covid19/cifib_dl.json",
    "--dataset_name", "covid19", \
    "--n_training_steps", "100000", \
    "--n_evaluation_steps", "2000", \
    "--n_report_steps", "2000", \
    "--b", "32", \
    "--n_warmup_steps", "20000", \
    "--model_name", "cifib", \
    "--model_config", "covid19/cifib.json",
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",\
]

earthquakes_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "generic_continuous", \
    "--dataloader_config", "earthquakes/cifib_dl.json",
    "--dataset_name", "earthquakes", \
    "--n_training_steps", "100000", \
    "--n_evaluation_steps", "2000", \
    "--n_report_steps", "2000", \
    "--b", "32", \
    "--n_warmup_steps", "20000", \
    "--model_name", "cifib", \
    "--model_config", "earthquakes/cifib.json",
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",\
]

syn_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "generic_continuous", \
    "--dataset_name", ["hawkes_1_continuous_v1", "hawkes_2_continuous_v1", "poisson_continuous_v1", "self_correct_continuous_v1", "stationary_renewal_continuous_v1"], \
    "--n_training_steps", "10000", \
    "--n_evaluation_steps", "2000", \
    "--n_report_steps", "500", \
    "--b", "128", \
    "--n_warmup_steps", "1000", \
    "--model_name", "cifib", \
    "--model_config", "syn/cifib.json",
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",\
]

citibike_plot_hyperparameter_list = [
    "train.py", \
    "--seed", "32", \
    "--model_name", "cifib", \
    "--model_config", "citibike/cifib.json", \
    "--lr", "0.002", \
    "--used_batch_size", "32", \
    "--n_training_steps", "100000", \
    "--dataset_name", "citibike", \
    "--dataloader_name", "generic_continuous", \
    "--figure_count", "10", \
    "--test", \
    "--used_dataloader_config", "cifib_dl.json", \
    "--plot_type", "intensity", \
    "--dataloader_config", "citibike/plot.json", \
    "--resolution", "200", \
    # "--task_name", ['mae_and_f1', 'mae_e_and_f1']
    "--task_name", ['mae_and_distance', 'mae_e_and_distance']
]

covid19_plot_hyperparameter_list = [
    "train.py", \
    "--seed", "32", \
    "--model_name", "cifib", \
    "--model_config", "covid19/cifib.json", \
    "--lr", "0.002", \
    "--used_batch_size", "32", \
    "--n_training_steps", "100000", \
    "--dataset_name", "covid19", \
    "--dataloader_name", "generic_continuous", \
    "--figure_count", "10", \
    "--test", \
    "--used_dataloader_config", "cifib_dl.json", \
    "--plot_type", "intensity", \
    "--dataloader_config", "covid19/plot.json", \
    "--resolution", "200", \
    "--task_name", ['mae_and_distance', 'mae_e_and_distance']
]

earthquakes_plot_hyperparameter_list = [
    "train.py", \
    "--seed", "32", \
    "--model_name", "cifib", \
    "--model_config", "earthquakes/cifib.json", \
    "--lr", "0.002", \
    "--used_batch_size", "32", \
    "--n_training_steps", "100000", \
    "--dataset_name", "earthquakes", \
    "--dataloader_name", "generic_continuous", \
    "--figure_count", "10", \
    "--test", \
    "--used_dataloader_config", "cifib_dl.json", \
    "--plot_type", "intensity", \
    "--dataloader_config", "earthquakes/plot.json", \
    "--resolution", "200", \
    "--task_name", ['mae_and_distance', 'mae_e_and_distance']
]

syn_plot_hyperparameter_list = [
    "train.py", \
    "--seed", "32", \
    "--model_name", "cifib", \
    "--model_config", "syn/cifib.json", \
    "--lr", "0.002", \
    "--used_batch_size", "128", \
    "--n_training_steps", "10000", \
    "--dataset_name", ["hawkes_1_continuous_v1", "hawkes_2_continuous_v1", "poisson_continuous_v1", "self_correct_continuous_v1", "stationary_renewal_continuous_v1"], \
    "--dataloader_name", "generic_continuous", \
    "--figure_count", "1", \
    # "--train", \
    "--test", \
    # "--evaluation", \
    # "--plot_type", ["intensity", "probability", "debug"], \
    # "--plot_type", ["intensity", "probability"], \
    "--plot_type", "probability", \
    "--dataloader_config", "syn/plot.json", \
    "--resolution", "200", \
    "--task_name", ['spearman_and_l1', 'graph']
]

training_hyperparameter = {
    'citibike': citibike_training_hyperparameter_list,
    'covid19': covid19_training_hyperparameter_list,
    'earthquakes': earthquakes_training_hyperparameter_list,
    'syn': syn_training_hyperparameter_list
}

plot_hyperparameter = {
    'citibike': citibike_plot_hyperparameter_list,
    'covid19': covid19_plot_hyperparameter_list,
    'earthquakes': earthquakes_plot_hyperparameter_list,
    'syn': syn_plot_hyperparameter_list
}
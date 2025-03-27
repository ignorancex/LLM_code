# parameter sets of model fenn

stackoverflow_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "generic", \
    "--dataloader_config", "stackoverflow/fenn_dl.json",
    "--dataset_name", "stackoverflow", \
    "--n_training_steps", "200000", \
    "--n_evaluation_steps", "2000", \
    "--n_report_steps", "2000", \
    "--b", "32", \
    "--n_warmup_steps", "40000", \
    "--model_name", "fenn", \
    "--model_config", "stackoverflow/fenn.json",
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",
]

retweet_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "generic", \
    "--dataloader_config", "retweet/fenn_dl.json",
    "--dataset_name", "retweet", \
    "--n_training_steps", "400000", \
    "--n_evaluation_steps", "4000", \
    "--n_report_steps", "4000", \
    "--b", "32", \
    "--n_warmup_steps", "80000", \
    "--model_name", "fenn", \
    "--model_config", "retweet/fenn.json",
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",
]

mooc_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "generic", \
    "--dataloader_config", "mooc/fenn_dl.json",
    "--dataset_name", "mooc", \
    "--n_training_steps", "400000", \
    "--n_evaluation_steps", "4000", \
    "--n_report_steps", "4000", \
    "--b", "32", \
    "--n_warmup_steps", "80000", \
    "--model_name", "fenn", \
    "--model_config", "mooc/fenn.json",
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",
]

bookorder_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "generic", \
    "--dataloader_config", "bookorder/fenn_dl.json",
    "--dataset_name", "bookorder", \
    "--n_training_steps", "20000", \
    "--n_evaluation_steps", "100", \
    "--n_report_steps", "100", \
    "--b", "8", \
    "--n_warmup_steps", "4000", \
    "--model_name", "fenn", \
    "--model_config", "bookorder/fenn.json",
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",
]

syn_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "generic", \
    "--dataset_name", ["hawkes_1_v2", "hawkes_2_v2", "poisson_v2", "self_correct_v2", "stationary_renewal_v2"], \
    "--n_training_steps", "10000", \
    "--n_evaluation_steps", "500", \
    "--n_report_steps", "500", \
    "--b", "128", \
    "--n_warmup_steps", "1000", \
    "--model_name", "fenn", \
    "--model_config", "syn/fenn.json",
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",
]

retweet_plot_hyperparameter_list = [
    "train.py", \
    "--seed", "32", \
    "--model_name", "fenn", \
    "--model_config", "retweet/fenn.json", \
    "--lr", "0.002", \
    "--used_batch_size", "32", \
    "--n_training_steps", "400000", \
    "--dataset_name", "retweet", \
    "--dataloader_name", "generic", \
    "--figure_count", "10", \
    # "--train", \
    # "--evaluation", \
    "--test", \
    "--used_dataloader_config", "fenn_dl.json", \
    # "--plot_type", ["intensity", "probability", "debug"], \
    "--plot_type", "intensity", \
    "--dataloader_config", "retweet/plot.json", \
    "--resolution", "200", \
    "--task_name", ['mae_and_f1', 'mae_e_and_f1']
]

stackoverflow_plot_hyperparameter_list = [
    "train.py", \
    "--seed", "32", \
    "--model_name", "fenn", \
    "--model_config", "stackoverflow/fenn.json", \
    "--lr", "0.002", \
    "--used_batch_size", "32", \
    "--n_training_steps", "200000", \
    "--dataset_name", "stackoverflow", \
    "--dataloader_name", "generic", \
    "--figure_count", "10", \
    # "--train", \
    # "--evaluation", \
    "--test", \
    "--used_dataloader_config", "fenn_dl.json", \
    # "--plot_type", ["intensity", "probability", "debug"], \
    "--plot_type", "intensity", \
    "--dataloader_config", "stackoverflow/plot.json", \
    "--resolution", "200", \
    "--task_name", ['mae_and_f1', 'mae_e_and_f1']
]

mooc_plot_hyperparameter_list = [
    "train.py", \
    "--seed", "32", \
    "--model_name", "fenn", \
    "--model_config", "mooc/fenn.json", \
    "--lr", "0.002", \
    "--used_batch_size", "32", \
    "--n_training_steps", "400000", \
    "--dataset_name", "mooc", \
    "--dataloader_name", "generic", \
    "--figure_count", "10", \
    # "--train", \
    # "--evaluation", \
    "--test", \
    "--used_dataloader_config", "fenn_dl.json", \
    # "--plot_type", ["intensity", "probability", "debug"], \
    "--plot_type", "intensity", \
    "--dataloader_config", "mooc/plot.json", \
    "--resolution", "200", \
    "--task_name", ['mae_and_f1', 'mae_e_and_f1']
]

bookorder_plot_hyperparameter_list = [
    "train.py", \
    "--seed", "32", \
    "--model_name", "fenn", \
    "--model_config", "bookorder/fenn.json", \
    "--lr", "0.002", \
    "--used_batch_size", "8", \
    "--n_training_steps", "20000", \
    "--dataset_name", "bookorder", \
    "--dataloader_name", "generic", \
    "--figure_count", "10", \
    # "--train", \
    # "--evaluation", \
    "--test", \
    "--used_dataloader_config", "fenn_dl.json", \
    # "--plot_type", ["intensity", "probability", "debug"], \
    "--plot_type", "intensity", \
    "--dataloader_config", "bookorder/plot.json", \
    "--resolution", "200", \
    "--task_name", ['mae_and_f1', 'mae_e_and_f1']
]

syn_plot_hyperparameter_list = [
    "train.py", \
    "--seed", "32", \
    "--model_name", "fenn", \
    "--model_config", "syn/fenn.json", \
    "--lr", "0.002", \
    "--used_batch_size", "128", \
    "--n_training_steps", "10000", \
    "--dataset_name", ["hawkes_1_v2", "hawkes_2_v2", "poisson_v2", "self_correct_v2", "stationary_renewal_v2"], \
    "--dataloader_name", "generic", \
    "--figure_count", "10", \
    "--train", \
    "--test", \
    "--evaluation", \
    # "--plot_type", ["intensity", "probability", "debug"], \
    "--plot_type", "intensity", \
    "--dataloader_config", "syn/plot.json", \
    "--resolution", "200", \
    "--task_name", 'spearman_and_l1'
]

training_hyperparameter = {
    'stackoverflow': stackoverflow_training_hyperparameter_list,
    'retweet': retweet_training_hyperparameter_list,
    'bookorder': bookorder_training_hyperparameter_list,
    'mooc': mooc_training_hyperparameter_list,
    'synthetic': syn_training_hyperparameter_list
}

plot_hyperparameter = {
    'retweet': retweet_plot_hyperparameter_list,
    'stackoverflow': stackoverflow_plot_hyperparameter_list,
    'bookorder': bookorder_plot_hyperparameter_list,
    'mooc': mooc_plot_hyperparameter_list,
    'synthetic': syn_plot_hyperparameter_list
}
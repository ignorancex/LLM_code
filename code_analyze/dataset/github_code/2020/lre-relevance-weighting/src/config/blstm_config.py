FEA_DIMENSION = 500

label_dict = {'alv': 0, 'fas': 1, 'prs': 2, 'pus': 3, 'urd': 4, 'mul': 5}

path_conf = {
    "dev_data_5s": '',
    "dev_labels_5s": '',
    "dev_data_15s": '',
    "dev_labels_15s": '',
    "train_data_15s": '',
    "train_labels_15s": '',
    "dev_list": '',
    "train_list": '',
    "key_file": ''
}

generate_config = {
    "train_files_per_labels": 20000,
    "dev_files_per_labels": 2000,
    "output_loc": ''
}

log_conf = {
    "base_dir": '',
    "file_name": '',
}

summary_conf = {
    "checkpoint_dir": '',
    "model_directory": '',
    "model_name": '',
}

model_conf = {
    "blstm_layer1_units": 256,
    "blstm_layer2_units": 256,
    "attention_size": 128,
    "dense_dim": 512,
    "num_of_classes": 6,
    "start_learning_rate": 1e-4,
    "decay_rate": 0.95,
    "decay_steps": 20000
}

train_conf = {
    "train_batch_size": 64,
    "val_batch_size": 70,
    "num_classes": 6,
    "num_epochs": 100,
    "num_batches": 1000,
    "window": 100,
    "hop": 20,
    "sample_len": 3010,
    "save_checkpoint": 150,
    "print_checkpoint": 50
}

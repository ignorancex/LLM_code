
import os


def get_original_dataset_file_path(dataset_dir_path, domain, dataset_split):
    return os.path.join(dataset_dir_path, f'{domain}_{dataset_split}.tsv')


def get_preprocessed_dataset_file_path(dataset_dir_path, domain, dataset_split):
    return os.path.join(dataset_dir_path, f'{domain}_{dataset_split}.jsonl')

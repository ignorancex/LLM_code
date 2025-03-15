"""
    Making 15s utterances for 1s i-vec blstm training, 5s and 15s for evaluation.
    Procedure: Sample 15s from large files in the ratio of number of files for each class to make the data-set balanced.
"""
from collections import Counter
from os.path import join
from tqdm import tqdm

import config.blstm_config as config
import numpy as np
import multiprocessing as mp

selection = 'dev'  # train|dev
duration = 5  # 5|15
window = config.train_conf['window'] / 100
hop = config.train_conf['hop'] / 100

one_sec_list = config.path_conf['{}_list'.format(selection)]
key_file = config.path_conf['key_file']
total_samples_per_label = config.generate_config['{}_files_per_label'.format(selection)]
num_samples = int((duration - window + hop) / hop + 1)
feature_dim = config.FEA_DIMENSION
save_prefix = join(config.generate_config['output_loc'], '{}_{}s_'.format(selection, duration))

one_sec_files = np.genfromtxt(one_sec_list, dtype=str)
file_to_label_dict = dict()
with open(key_file) as f:
    for line in f.readlines()[1:]:
        tokens = line.strip().split()
        file_to_label_dict[tokens[0]] = tokens[3]

labels_dict = config.label_dict
labels = labels_dict.keys()


def get_label(file_name):
    base_name = file_name.strip().rsplit('/', 1)[1].rsplit('_', 1)[0]
    label = file_to_label_dict[base_name]
    try:
        _ = labels_dict[label]
        return label
    except KeyError:
        return 'mul'


def run_parallel(func, args_list, n_workers=10, p_bar=True):
    pool = mp.Pool(n_workers)
    if p_bar:
        if type(args_list) is list:
            total_len = len(args_list)
        else:
            total_len = args_list.shape[0]
        out = tqdm(pool.imap(func, args_list), total=total_len)
    else:
        out = pool.map(func, args_list)
    pool.close()
    # pool.join()
    if out is not None:
        return list(out)


def get_sample(file_name):
    sample_data = np.load(file_name)
    if sample_data.shape[-1] > num_samples:
        start_loc = np.random.choice(sample_data.shape[-1] - num_samples, 1)[0]
        sample = sample_data[:, start_loc: start_loc + num_samples].T
        return sample
    return None


full_labels = list(file_to_label_dict.values())
full_counter = Counter(full_labels)
print(full_counter)
one_sec_labels = np.array([get_label(one_file) for one_file in one_sec_files])
labels_counter = Counter(one_sec_labels)
print(labels_counter)

one_sec_files_for_labels = dict([(label, one_sec_files[one_sec_labels == label]) for label in labels])

print('Sampling the files.')
total_samples = total_samples_per_label * len(labels)
sampled_files = np.array([np.random.choice(one_sec_files_for_labels[label], total_samples_per_label) for label in labels]).reshape([-1, ])
sampled_labels = np.array([[label] * total_samples_per_label for label in labels]).reshape([-1, ])


sampled_data = run_parallel(get_sample, sampled_files, n_workers=25)
ignore_idx = [idx for idx, data in enumerate(sampled_data) if data is None]
sampled_data = np.array(sampled_data)

if len(ignore_idx) > 0:
    sampled_data = np.stack(np.delete(sampled_data, ignore_idx, axis=0))
    sampled_labels = np.delete(sampled_labels, ignore_idx, axis=0)


print('Sampled {} files.'.format(sampled_data.shape[0]))
print(sampled_data.shape, sampled_labels.shape)
print('Saving to disk.')
np.save(save_prefix + 'data.npy', np.array(sampled_data))
np.save(save_prefix + 'labels.npy', np.array(sampled_labels))

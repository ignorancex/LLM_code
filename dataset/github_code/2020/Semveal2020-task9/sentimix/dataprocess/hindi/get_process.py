import os
import logging
import sys
import pickle

from utils import process_text, write_file, write_as_test,process_test


train_file = os.path.join('data', 'train_14k_split_conll.txt')
train_texts, train_labels, train_sid = process_text(train_file)
print(len(train_texts), len(train_labels))
print(train_texts)
print(train_labels)
# train_texts = clean_str(train_texts)

train_write_file = os.path.join('data', 'get_train_f.tsv')
write_file(train_write_file, train_texts, train_labels)

trial_file = os.path.join('data', 'dev_3k_split_conll.txt')
trial_texts, trial_labels, trial_sid = process_text(trial_file)
print(len(trial_texts), len(trial_labels))
# trial_texts = clean_str(trial_texts)

trial_write_file = os.path.join('data', 'get_dev_f.tsv')
write_file(trial_write_file, trial_texts, trial_labels)

process_test()






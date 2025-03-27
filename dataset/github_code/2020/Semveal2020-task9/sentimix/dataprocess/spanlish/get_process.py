import os
import logging
import sys
import pickle

from utils import process_text, write_file, write_as_test,process_test


train_file = os.path.join('Data', 'train.conll')
train_texts, train_labels, train_sid = process_text(train_file)
# print(len(train_texts), len(train_labels))
# print(train_texts)
# print(train_labels)


train_write_file = os.path.join('Data', 'get_train.tsv')
write_file(train_write_file, train_texts, train_labels)

trial_file = os.path.join('Data', 'dev.conll')
trial_texts, trial_labels, trial_sid = process_text(trial_file)
print(len(trial_texts), len(trial_labels))

trial_write_file = os.path.join('Data', 'get_dev.tsv')
write_file(trial_write_file, trial_texts, trial_labels)
#
process_test()






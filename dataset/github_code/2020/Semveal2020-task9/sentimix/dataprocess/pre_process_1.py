import os
import logging
import sys
import pickle

from utils import process_text, write_file, write_as_test, clean_str


train_file = os.path.join('data', 'train.txt')
train_texts, train_labels, train_sid = process_text(train_file)
print(len(train_texts), len(train_labels))
train_texts = clean_str(train_texts)

train_write_file = os.path.join('data', 'train.tsv')
write_file(train_write_file, train_texts, train_labels)

trial_file = os.path.join('data', 'trial_conll.txt')
trial_texts, trial_labels, trial_sid = process_text(trial_file)
print(len(trial_texts), len(trial_labels))
trial_texts = clean_str(trial_texts)

trial_write_file = os.path.join('data', 'dev.tsv')
write_file(trial_write_file, trial_texts, trial_labels)

test_file = os.path.join('data', 'test.tsv')
write_as_test(test_file, trial_texts, trial_sid)


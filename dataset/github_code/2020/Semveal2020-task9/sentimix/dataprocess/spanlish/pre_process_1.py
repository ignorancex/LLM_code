import os
import logging
import sys
import pickle

from utils import process_text, write_file, write_as_test,clean_str


train_file = os.path.join('Data', 'train_conll_spanglish.txt')
train_texts, train_labels, train_sid = process_text(train_file)
train_texts = clean_str(train_texts)


train_write_file = os.path.join('Data_ne', 'train_org_sapnlish_ne_unk.tsv')
write_file(train_write_file, train_texts, train_labels)

# trial_file = os.path.join('Data', 'trail_conll_spanglish.txt')
# trial_texts, trial_labels, trial_sid = process_text(trial_file)
# print(len(trial_texts), len(trial_labels))
# trial_texts = clean_str(trial_texts)
#
# trial_write_file = os.path.join('Data_ne', 'dev_org_spanglish_ne.tsv')
# write_file(trial_write_file, trial_texts, trial_labels)
#
# test_file = os.path.join('Data_ne', 'test_org_spanglish.tsv')
# write_as_test(test_file, trial_texts, trial_sid)


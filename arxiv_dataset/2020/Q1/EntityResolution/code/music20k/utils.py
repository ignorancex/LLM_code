from future.builtins import next

import os
import csv
import json
import re
import logging
import optparse
import collections
import itertools
import numpy as np
import random
from dedupe.core import randomPairs, randomPairsMatch

from unidecode import unidecode

import scipy.io

def preProcess(column):
    """
    Do a little bit of data cleaning with the help of Unidecode and Regex.
    Things like casing, extra spaces, quotes and new lines can be ignored.
    """
    try : # python 2/3 string differences
        column = column.decode('utf8')
    except AttributeError:
        pass
    column = unidecode(column)
    column = re.sub("[()]", '', column)
    column = re.sub(",", '', column)
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()
    # If data is missing, indicate that by setting the value to `None`
    if not column:
        column = None
    return column

def readData(input_file, set_delim='**'):
    data_d = {}
    label_d = {}
    with open(input_file) as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            row = dict((k, preProcess(v)) for k, v in row.items())
            data_d[int(row['TID'])] = row
            label_d[int(row['TID'])] = int(row['CID'])
    fieldnames = reader.fieldnames
    return data_d, label_d, fieldnames

def readGoldData(input_file, golden_file, set_delim='**'):            
    label_d = {}
    data_d = {}
    with open(golden_file) as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            label_d[int(row['TID'])] = int(row['True Id'])
            data_d[int(row['TID'])] = True

    with open(input_file) as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if int(row['TID']) not in data_d:
                continue
            row = dict((k, preProcess(v)) for k, v in row.items())
            data_d[int(row['TID'])] = row

    fieldnames = reader.fieldnames
    
    return data_d, label_d, fieldnames


def makeTrainingPairs(data_d, label_d, sample_ratio=1.0, seed = 42):
    identified_records = collections.defaultdict(list)
    matched_pairs = set()
    distinct_pairs = set()
    unique_record_ids = set()

    
    #distinct_size = int(0.1 * sample_ratio * len(data_d) * len(data_d))
    distinct_size = int(0.00001 *  len(data_d) * len(data_d))
    
    # a list of record_ids associated with each common_key
    for record_id, record in data_d.items():
        unique_record_ids.add(record_id)
        identified_records[label_d[record_id]].append(record_id)
        
    # all combinations of matched_pairs from each common_key group
    for record_ids in identified_records.values():
        if len(record_ids) > 1:
            matched_pairs.update(itertools.combinations(sorted(record_ids), 2))
            
    # calculate indices using dedupe.core.randomPairs to avoid
    # the memory cost of enumerating all possible pairs
    unique_record_ids = list(unique_record_ids)
    pair_indices = randomPairs(len(unique_record_ids),
                               distinct_size,
                               seed)
    distinct_pairs = set()
    for i, j in pair_indices:
        distinct_pairs.add((unique_record_ids[i],
                            unique_record_ids[j]))

    distinct_pairs -= matched_pairs

    #match_size = int(sample_ratio * len(matched_pairs))
    match_size = int(0.2 * len(matched_pairs))
    matched_pairs = set(random.sample(matched_pairs, match_size))
    
    matched_records = [(data_d[key_1], data_d[key_2])
                       for key_1, key_2 in matched_pairs]

    distinct_records = [(data_d[key_1], data_d[key_2])
                        for key_1, key_2 in distinct_pairs]
    
    training_pairs = {'match': matched_records,
                      'distinct': distinct_records}

    return training_pairs
    


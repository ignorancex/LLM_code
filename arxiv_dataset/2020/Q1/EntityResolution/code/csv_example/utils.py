from future.builtins import next

import os
import csv
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
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()
    # If data is missing, indicate that by setting the value to `None`
    if not column:
        column = None
    return column

def readData(filename):
    """
    Read in our data from a CSV file and create a dictionary of records, 
    where the key is a unique record ID and each value is dict
    """

    data_d = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = [(k, preProcess(v)) for (k, v) in row.items()]
            row_id = int(row['Id'])
            data_d[row_id] = dict(clean_row)

    return data_d

def readGoldData(filename):
    """
    Read in our data from a CSV file and create a dictionary of records, 
    where the key is a unique record ID and each value is dict
    """

    data_d = {}
    label_d = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = [(k, preProcess(v)) for (k, v) in row.items()]
            data_d[int(row['Id'])] = dict(clean_row)
            label_d[int(row['Id'])] = row['True Id']
    return data_d, label_d, reader.fieldnames


def makeTrainingPairs(data_d, label_d, sample_ratio=1.0, seed = 42):
    identified_records = collections.defaultdict(list)
    matched_pairs = set()
    distinct_pairs = set()
    unique_record_ids = set()

    #distinct_size = int(0.1 * sample_ratio * len(data_d) * len(data_d))
    distinct_size = int(0.1 *  len(data_d) * len(data_d))

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
    match_size = int(1.0 * len(matched_pairs))
    matched_pairs = set(random.sample(matched_pairs, match_size))

    matched_records = [(data_d[key_1], data_d[key_2])
                       for key_1, key_2 in matched_pairs]

    distinct_records = [(data_d[key_1], data_d[key_2])
                        for key_1, key_2 in distinct_pairs]
    
    training_pairs = {'match': matched_records,
                      'distinct': distinct_records}

    return training_pairs
    


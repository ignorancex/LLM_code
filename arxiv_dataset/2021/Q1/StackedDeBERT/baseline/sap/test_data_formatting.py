import csv
from baseline.base_utils import INTENTION_TAGS

''' Convert .tsv to .csv without header for each intent
Format:
example1;intent1
example2;intent1
...
exampleN;intentM
'''

complete = False
perc = 0.8
dataset_arr = ['snips']

for dataset in dataset_arr:
    tags = INTENTION_TAGS[dataset]

    # Data dir path
    data_dir_path = "../../data/snips_intent_data/"
    if complete:
        data_dir_path += "complete_data/test.tsv"
    else:
        data_dir_path += "comp_with_incomplete_data_tfidf_lower_{}_noMissingTag/test.tsv".format(perc)

    tsv_file = open(data_dir_path)
    reader = csv.reader(tsv_file, delimiter='\t')

    # Write csv
    results_dir_path = data_dir_path.split('.tsv')[0] + "_sap.csv"
    file_test = open(results_dir_path, 'wt')
    dict_writer = csv.writer(file_test, delimiter=',')

    row_count = 0
    sentences, intents = [], []
    for row in reader:
        if row_count != 0:
            dict_writer.writerow([row[0], row[1]])
        row_count += 1

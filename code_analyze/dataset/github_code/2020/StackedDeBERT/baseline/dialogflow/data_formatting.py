import csv
from baseline.base_utils import INTENTION_TAGS
from collections import defaultdict

''' Convert .tsv to .csv without header for each intent
Format:
example1
example2
...
exampleN
'''

complete = False
perc = 0.8
dataset_arr = ['snips']

for dataset in dataset_arr:
    tags = INTENTION_TAGS[dataset]

    for type in ['train', 'test']:
        # Data dir path
        data_dir_path = "../../data/snips_intent_data/"
        if complete:
            data_dir_path += "complete_data/"
        else:
            data_dir_path += "incomplete_data_noTag/comp_with_incomplete_data_tfidf_lower_{}_noMissingTag/{}.tsv".format(perc, type)

        # Read tsv
        tsv_file = open(data_dir_path)
        reader = csv.reader(tsv_file, delimiter='\t')

        dict_intents = defaultdict(lambda: [])
        row_count = 0
        sentences = []
        for row in reader:
            if row_count != 0:
                dict_intents[tags[row[1]]].append(row[0])
            row_count += 1

        for k, v in dict_intents.items():
            # Write csv
            results_dir_path = data_dir_path.split('.tsv')[0] + "_dialogflow_{}.csv".format(k)
            file_test = open(results_dir_path, 'wt')
            dict_writer = csv.writer(file_test, delimiter=';')

            for text in dict_intents[k]:
                dict_writer.writerow([text])

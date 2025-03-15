import csv
from baseline.base_utils import INTENTION_TAGS, INTENTION_TAGS_WITH_SPACE

''' Convert .tsv to .csv without header for each intent
Format:
example1 intent1
example2 intent1
...
exampleN intentM
'''

complete = False
dataset_arr = ['snips']

for perc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.8]:
    for dataset in dataset_arr:
        tags = INTENTION_TAGS_WITH_SPACE[dataset]

        for type in ['test', 'train']:
            # Data dir path
            data_dir_path = "../../data/snips_intent_data/"
            if complete:
                data_dir_path += "complete_data/"
            else:
                data_dir_path += "comp_with_incomplete_data_tfidf_lower_{}_noMissingTag/".format(perc)
            data_dir_path += "{}.tsv".format(type)

            tsv_file = open(data_dir_path)
            reader = csv.reader(tsv_file, delimiter='\t')

            # Write csv
            results_dir_path = data_dir_path.split('.tsv')[0] + "_semantic_hashing.csv"
            file_test = open(results_dir_path, 'wt')
            dict_writer = csv.writer(file_test, delimiter='\t')

            row_count = 0
            sentences, intents = [], []
            for row in reader:
                if row_count != 0:
                    dict_writer.writerow([row[0], tags[row[1]]])
                row_count += 1

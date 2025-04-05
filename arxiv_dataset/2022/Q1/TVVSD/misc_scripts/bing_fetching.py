import os
import pandas as pd
import requests


queries = pd.read_csv('sense_specific_search_engine_queries.tsv', sep='\t', dtype={'sense_num': str})
queries['sense_num'] = queries['sense_num'].apply(lambda x: x.replace('.', '_'))
queries['queries'] = queries['queries'].apply(lambda s: s.split(','))

DOWNLOAD_FOLDER = 'bing_download/'

for i, row in enumerate(queries.itertuples()):
    verb = getattr(row, 'verb')
    sense_num = getattr(row, 'sense_num')
    sense_directory = DOWNLOAD_FOLDER + verb + '_' + sense_num + '/'

    query_list = getattr(row, 'queries')

    try:  
        os.mkdir(sense_directory)
    except OSError:  
        print ("Creation of the directory %s failed" % DOWNLOAD_FOLDER)
    else:  
        print ("Successfully created the directory %s " % DOWNLOAD_FOLDER)

    for i, query in enumerate(query_list):
        query_directory = sense_directory + 'q' + str(i)
        os.mkdir(query_directory)

        os.system("./bbid.py -s '" + query.strip() + "' -o " + query_directory + " --limit 500 --adult-filter-on --threads=5")

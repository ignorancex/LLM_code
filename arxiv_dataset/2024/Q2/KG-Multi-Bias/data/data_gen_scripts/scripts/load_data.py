import csv 
import json
import pickle
from eventregistry import *

def create_datafiles(apiKey, source_lst):

    url_endpoint = 'https://eventregistry.org/api/v1/article/getArticles'
    url_params = {  
                    'resultType': 'articles',
                    'articleBodyLen': 0,
                    'articleUri': '', 
                    'includeArticleBody' : 'false',
                    'includeArticleEventUri' : 'false',
                    'includeArticleCategories' : 'true',
                    'apiKey' : ''
                }
    #REST API call to get the article details

    #csv file 
    csvfile = open('data/raw/dataset.csv', 'a', newline='\n')

    writer = csv.writer(csvfile, delimiter=',')

    # fieldnames = ['source','title','lang']

    # writer.writerow(fieldnames)

    with open("data/intermediate/uri_dict.txt", "rb") as myFile:
        uri_dict = pickle.load(myFile)

    csv_file = open('data/raw/dataset.csv','r',newline='\n') 
    csv_reader = csv.reader(csv_file, delimiter=',')

    print(len(uri_dict))

    for row in csv_reader:
        try:
            del uri_dict[row[3]]
        except:
            pass
        
    print(len(uri_dict))
    

    for uri in uri_dict:

        try:

            url_params['articleUri'] = uri
            url_params['apiKey'] = apiKey

            response = requests.get(url_endpoint, params=url_params)
            y = json.loads(response.text)

            row_text = []
            row_text.append(y['articles']['results'][0]['source']['uri'].split('.')[0])
            row_text.append(y['articles']['results'][0]['title'])
            row_text.append(y['articles']['results'][0]['lang'])
            writer.writerow(row_text)
        except:
            pass
            print('-------------')
            print(uri)

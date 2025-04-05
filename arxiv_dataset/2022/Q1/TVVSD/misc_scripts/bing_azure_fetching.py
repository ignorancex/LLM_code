import hashlib
import os
import pandas as pd
import posixpath
import requests
import urllib.parse
import urllib3
from azure.cognitiveservices.search.imagesearch import ImageSearchAPI
from msrest.authentication import CognitiveServicesCredentials


def fetch_query_images(search_term, download_folder):
    image_results = client.images.search(query=search_term, count=500, safe_search='Strict')
    hashes = []

    if image_results.value:
        with open(download_folder + '/urls.txt', 'w') as urls_file:
            for index, image in enumerate(image_results.value):
                urls_file.write(str(index) + ': ' + image.content_url +'\n')
                try:
                    r = requests.get(image.content_url, timeout=15, verify=False)
                except requests.exceptions.RequestException as exc:
                    print(exc)
                    continue
                if r.status_code == 200:
                    remote_path = urllib.parse.urlsplit(image.content_url).path
                    #Strip GET parameters from filename
                    filename = posixpath.basename(remote_path).split('?')[0]
                    name, ext = os.path.splitext(filename)
                    name = name[:36]
                    filename = str(index) + '_' + name + ext

                    md5_key = hashlib.md5(r.content).hexdigest()
                    if md5_key not in hashes:
                        hashes.append(md5_key)

                        with open(download_folder + '/'+ filename, 'wb') as downloaded_files:
                            downloaded_files.write(r.content)
                            downloaded_files.close()
            urls_file.close()
    else:
        raise RuntimeError("No image results returned for query %s" % search_term)


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
queries = pd.read_csv('sense_specific_search_engine_queries.tsv', sep='\t',
                      dtype={'sense_num': str})
queries['sense_num'] = queries['sense_num'].apply(lambda x: x.replace('.', '_'))
queries['queries'] = queries['queries'].apply(lambda s: s.split(','))

subscription_key = "INSERT AZURE KEY HERE" 
DOWNLOAD_FOLDER = 'bing_download_azure/'
client = ImageSearchAPI(CognitiveServicesCredentials(subscription_key))


for _, row in enumerate(queries.itertuples()):
    verb = getattr(row, 'verb')
    sense_num = getattr(row, 'sense_num')
    sense_directory = DOWNLOAD_FOLDER + verb + '_' + sense_num + '/'
    print(sense_directory)

    query_list = getattr(row, 'queries')

    try:
        os.mkdir(sense_directory)
    except OSError:
        print("Creation of the directory %s failed" % DOWNLOAD_FOLDER)

    for i, query in enumerate(query_list):
        query_directory = sense_directory + 'q' + str(i)
        try:
            os.mkdir(query_directory)
        except OSError:
            print("Creation of the directory %s failed" % query_directory)
        fetch_query_images(query.strip(), query_directory)

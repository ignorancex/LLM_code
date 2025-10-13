from os.path import join, splitext, basename
import pickle

import numpy as np
import pandas as pd

from yottixel_code.utils import BoB, load_mosaic
from yottixel_code.consts import SITES_DICT, DIAGNOSES_DICT, VIEW_URL


def main_search(query_slide_path, save_dir, database_dir, database_features_path, metadata_path, query_extension, k, subtype_search, query_site):
    query_features_path = join(save_dir, splitext(basename(query_slide_path))[0], 'features.pkl')
    
    with open(database_features_path, 'rb') as f:
        database_features = pickle.load(f)
    
    with open(query_features_path, 'rb') as f:
        query_features = pickle.load(f)

    metadata = pd.read_csv(metadata_path)
    metadata = metadata.set_index('file_name')

    if subtype_search:
        assert query_site is not None

    # Creating database Bobs
    database_features_dict = {}
    for fname, feature in database_features.items():
        file_name = fname.split("_patch")[0] + ".svs"
        if subtype_search:
            assert query_site is not None
            
            if SITES_DICT[metadata.loc[file_name, "primary_site"]] == query_site:
                database_features_dict.setdefault(file_name, []).extend([feature])
        else:
            database_features_dict.setdefault(file_name, []).extend([feature])
        
    database_BoBs = []
    for file_name, feature_queue in database_features_dict.items():
        barcodes = (np.diff(np.array(feature_queue), n=1, axis=1) < 0) * 1
        if file_name in metadata.index:
            site = SITES_DICT[metadata.loc[file_name, "primary_site"]]
            diagnosis = DIAGNOSES_DICT[metadata.loc[file_name, 'project_name']]
        else:
            site = None
            diagnosis = None
        database_BoBs.append(BoB(barcodes, file_name, site, diagnosis))

    # Creating query Bob
    file_name = f"{list(query_features.keys())[0].split('_patch')[0]}.{query_extension}"
    feature_queue = [feature for feature in query_features.values()]
    barcodes = (np.diff(np.array(feature_queue), n=1, axis=1) < 0) * 1
    query_BoB = BoB(barcodes, file_name, 'query', 'query')

    # Calculating the results
    distances = sorted([(bob.distance(query_BoB), bob) for bob in database_BoBs], key=lambda x: x[0])
    results = distances[:k]

    # Saving as pandas dataframe    
    save_path = join(save_dir, splitext(basename(query_slide_path))[0], 'results.csv')
    print(f"Search completed. Saving results to {save_path}.")
    
    names = [splitext(bob.file_name)[0] for _, bob in results]
    sites = [splitext(bob.site)[0] for _, bob in results]
    diags = [splitext(bob.diagnosis)[0] for _, bob in results]
    dists = [dist for dist, _ in results]
    paths = [join( bob.site, bob.diagnosis, bob.file_name) for _, bob in results]
    links = [VIEW_URL + metadata.loc[bob.file_name, "id"] for _, bob in results]
    df_dict = {
        'Retreived Slide': names,
        'Distance': dists,
        'Site': sites,
        'Diagnosis': diags,
        'Path': paths,
        'Link': links,
    } 
    df = pd.DataFrame(df_dict)
    df.to_csv(save_path, index=False)
    return paths

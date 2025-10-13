'''
Created on Apr 25, 2024

@author: bernardo
'''

import os
import requests
from tqdm import tqdm
import pandas as pd
import networkx as nx
import pycountry_convert as pc
from .flights_dataset_utils import continents_colors

def load_football_dataset(football_data_path, initial_year=1872, final_year=2024, weighted=True):    
    if os.path.isdir(os.path.dirname(football_data_path)):
        if not os.path.exists(football_data_path):
            download_football_dataset(football_data_path)
    else:
        raise Exception("Provided path for football dataset is not reachable")
    
    df_matches = pd.read_csv(football_data_path, parse_dates=["Date"], date_format='%m/%d/%Y')
    df_matches['year'] = df_matches.Date.dt.year
    # Keep only desired years
    df_matches = df_matches[df_matches.year>=initial_year]
    df_matches = df_matches[df_matches.year<=final_year]
    
    
    # Count how many matches they played regardless of who was home or away.
    ordered_matches = pd.DataFrame(df_matches[['HomeTeam','AwayTeam']].apply(lambda x: sorted([x['HomeTeam'],x['AwayTeam']]),axis=1).tolist())
    ordered_matches.columns = ['team1','team2']
    num_matchs = ordered_matches.groupby(['team1','team2']).size().reset_index()
    num_matchs.columns = ['team1','team2','weight']
    
    
    if weighted:
        G = nx.from_pandas_edgelist(num_matchs,source='team1',target='team2',edge_attr='weight',create_using=nx.Graph())
    else: 
        G = nx.from_pandas_edgelist(num_matchs,source='team1',target='team2',create_using=nx.Graph())
            
    countries_list = pd.unique(num_matchs[['team1', 'team2']].values.ravel('K'))
    
    # Add atributes to graph's nodes
    continents_dict = get_continents_dicts(countries_list)
    colors_dict = {country: continents_colors[continent] for country, continent in continents_dict.items()}
    nx.set_node_attributes(G, continents_dict, name='continent')
    nx.set_node_attributes(G, colors_dict, name='color')
    
    return G

def get_continents_dicts(countries_list):
    continents_dict = {}
    for country in countries_list: 
        try:
            continent_code = pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country))
            continents_dict[country] = pc.convert_continent_code_to_continent_name(continent_code)
        except:
            continue
            
    continents_dict['Curacao']= 'South America'
    continents_dict['Dem. Rep. of Congo'] = 'Africa'
    continents_dict['East Germany'] = 'Europe'
    continents_dict['East Timor'] = 'Asia'
    continents_dict['England'] = 'Europe'
    continents_dict['Khmer'] = 'Asia'
    continents_dict['Kosovo'] = 'Europe'
    continents_dict['New Hebrides'] = 'Oceania'
    continents_dict['Northern Ireland'] = 'Europe'
    continents_dict['Reunion'] = 'Africa'
    continents_dict['Saar'] = 'Europe'
    continents_dict['Saint Barthelemy'] = 'North America'
    continents_dict['Sao Tome e Principe'] = 'Africa'
    continents_dict['Scotland'] = 'Europe'
    continents_dict['Sint Maarten'] = 'South America'
    continents_dict['South Yemen'] = 'Asia'
    continents_dict['St. Vincent & Grenadines'] = 'South America'
    continents_dict['Surinam'] = 'South America'
    continents_dict['Tahiti'] = 'Oceania'
    continents_dict['Zanzibar'] = 'Africa'
    continents_dict['Wales'] = 'Europe'
    continents_dict['US Virgin Islands'] = 'North America'
    continents_dict['Tibet'] = 'Asia'
    continents_dict['Vatican'] = 'Europe'
    
    return continents_dict


def create_football_graphs(df_matches, weighted=True):
    all_graphs = []
    
    years = df_matches.year.unique()
    
        
    for year in years:
        df_matches_year = df_matches[df_matches.year==year]
        
        # Count how many matches they played regardless of who was home or away.
        ordered_matches = pd.DataFrame(df_matches_year[['HomeTeam','AwayTeam']].apply(lambda x: sorted([x['HomeTeam'],x['AwayTeam']]),axis=1).tolist())
        ordered_matches.columns = ['team1','team2']
        num_matchs = ordered_matches.groupby(['team1','team2']).size().reset_index()
        num_matchs.columns = ['team1','team2','weight']
        
        # two version of the graph: weighted and unweighted
        if weighted:
            G = nx.from_pandas_edgelist(num_matchs,source='team1',target='team2',edge_attr='weight',create_using=nx.Graph())
        else: 
            G = nx.from_pandas_edgelist(num_matchs,source='team1',target='team2',create_using=nx.Graph())
        
        all_graphs.append(G)
        
    return all_graphs, years

def download_football_dataset(matches_filename='AllMatches.csv', matches_url='https://www.fing.edu.uy/owncloud/index.php/s/V2tk4MxZxAvNidx/download',
                              countries_filename='countries_codes.csv', countries_codes_url='https://www.fing.edu.uy/owncloud/index.php/s/gb7SIbuUzVOgQJj/download',
                              map_figure_filename='./figures/world.jpg', map_figure_url='https://www.fing.edu.uy/owncloud/index.php/s/ABbYDE4nR4T3qvx/download'):
    # Code from https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
    response = requests.get(matches_url, stream=True)
    total_size = int(response.headers.get('content-length',0))
    with open(matches_filename, "wb") as f, tqdm(desc='Downloading football dataset', total=total_size, unit='B', unit_divisor=1024, unit_scale=True) as pbar:
        for football_data in response.iter_content(chunk_size=1024):
            size = f.write(football_data)
            pbar.update(size)
            
    response = requests.get(countries_codes_url, stream=True)
    total_size = int(response.headers.get('content-length',0))
    
    with open(os.path.join(os.path.dirname(matches_filename), countries_filename), "wb") as f, tqdm(desc='Downloading countries codes dataset', total=total_size, unit='B', unit_divisor=1024, unit_scale=True) as pbar:
        for countries_data in response.iter_content(chunk_size=1024):
            size = f.write(countries_data)
            pbar.update(size)
            
    response = requests.get(map_figure_url, stream=True)
    total_size = int(response.headers.get('content-length',0))
    with open(map_figure_filename, "wb") as f, tqdm(desc='Downloading world map figure', total=total_size, unit='B', unit_divisor=1024, unit_scale=True) as pbar:
        for figure_data in response.iter_content(chunk_size=1024):
            size = f.write(figure_data)
            pbar.update(size)
            
from sklearn.utils import check_random_state 
from numpy.random import Generator, PCG64
from pathlib import Path
from pandas import read_csv
from get_contexts import get_word_embedding

data_dir = '../Data/'
dataset = {
            'set1':{
                'name'  :   'semanticscholar',
                'f_name'    :   'data.tsv',
                'delimiter' :   '\t',   #it is a tab seperated file
                'column_names'  :   ['session_id', 'time', 'query_text', 'top_results', 'click'],  # full column
                'use_columns'  :   ['session_id', 'time', 'query_text'],   #for the time being we consider only this three column
                'processed_columns'  :   ['session_id', 'time', 'query_text', 'hashed'],   #for the time being we consider only this three column
                'row_skip'  :   0,
                },
            }

def get_use_columns(dt):
    return dataset[dt]['use_columns']

def get_processed_columns(dt):
    return dataset[dt]['processed_columns']

def get_data_source(dt):
    return dataset[dt]['name']

def get_data_file(dt):
    return dataset[dt]['f_name']

def get_delimiter(dt):
    return dataset[dt]['delimiter']

def get_column_names(dt):
    return dataset[dt]['column_names']

def get_rows_to_skip(dt):
    return dataset[dt]['row_skip']

def load_source_data(dt):
    data_source = get_data_source(dt)
    source_file = get_data_file(dt)
    source_file = Path(data_dir,data_source,source_file)
    print("reading from %s " %source_file)
    delm = get_delimiter(dt)
    cols = get_column_names(dt)
    use_cols = get_use_columns(dt)
    skr = get_rows_to_skip(dt)
    df = read_csv(source_file,sep=delm,names=cols,usecols=use_cols,skiprows=skr,header=None)
    return df

def load_processed_data(dt,d_file):
    delm = get_delimiter(dt)
    cols = get_processed_columns(dt)
    print("reading from %s " %d_file)
    df = read_csv(d_file,sep=delm,names=cols,header=None,dtype=object)
    return df

def get_dest_file(dt):
    data_source = get_data_source(dt)
    source_file = get_data_file(dt)
    dest_file = 'processed_'+source_file
    dest_file = Path(data_dir,data_source,dest_file)
    return dest_file

def get_data(dt):
    #candidate_set_sz = 500
    #epsilon = 0.5
    rg = Generator(PCG64(12345))
    #cnt = 100
    cnt = 5
    dest_file = get_dest_file(dt)
    if dest_file.is_file():
        df = load_processed_data(dt,dest_file)
        #actions = df.index.values
        #Qm = df.query_text.to_numpy()
        #X = get_word_embedding(df.query_text.to_numpy()).detach().cpu().numpy()
        X = df.query_text.to_numpy()
        anchors = rg.choice(df['session_id'].unique(),cnt,replace=False)
        s = df.drop_duplicates('session_id')
        anchor_ids = s[s['session_id'].isin(anchors)].index.values #first index of the session
        #anchor_features = X[anchor_ids,:]
        
        return df, X, anchor_ids, cnt
        #return df, X, actions, anchor_ids, anchor_features, cnt
        '''for i in anchor_ids:
            cos_sim = cosine_similarity(X[i,:].reshape(1,-1),X)
            #print(np.delete(actions,i))
            #print(np.delete(cos_sim.ravel(),i))
            #action indices are removed by value whereas similarity indices are removed by index
            arms = submodular_select(actions[actions != i],np.delete(cos_sim.ravel(),i),candidate_set_sz,epsilon)
        '''

# NOTE may need to be fixed as the file paths are hardcoded (and it was moved)

import pandas as pd

import sys
from pathlib import Path

from src.utils.cleaning import *  

DATA_PATH = Path('C:\\Users\\rooty\\UWEC\\Research\\CyberBullyingML\\venv\\cyberbullying-ml\\data\\original\\')
CLEAN_DATA_PATH = Path('C:\\Users\\rooty\\UWEC\\Research\\CyberBullyingML\\venv\\cyberbullying-ml\\data\\en_only\\')

def clean(text):
    #text = lowercase(text)    
    #text = remove_broken_links(text) # This was used for onlineHarassmentDataset 
    text = remove_links(text)
    text = remove_mentions(text)
    text = remove_rt(text)
    text = remove_html_entities(text)
    text = remove_non_english(text)
    text = decontract(text)
    text = remove_punctuation(text)
    #text = remove_digits(text)
    #text = lemmatize(text)
    #text = stem(text)
    #text = remove_stop_words(text)
    text = remove_excess_spaces(text)
    text = remove_space_at_beginning_or_end(text)
    text = remove_short_tweets(text)  
    return text

if __name__ == '__main__':
    
    filename = sys.argv[1] 
    text_col = sys.argv[2]
    encoding = sys.argv[3]

    file_path = DATA_PATH / filename

    df = pd.read_csv(file_path, encoding_errors='ignore')
    
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    df.loc[:, text_col] = df.loc[:, text_col].apply(clean) # clean tweets
    
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    f = Path(filename)
    df.to_csv(CLEAN_DATA_PATH / (f.stem + '_basic_clean' + f.suffix), encoding=encoding, index=False)
    
    
    


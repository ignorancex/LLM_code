from data import *
from pathlib import Path
#from pandas import read_csv
from urllib.parse import unquote
import json
import numpy as np
import hashlib

data_dir = '../Data/'

'''def get_dest_file(dt):
    data_source = get_data_source(dt)
    source_file = get_data_file(dt)
    dest_file = 'processed_'+source_file
    dest_file = Path(data_dir,data_source,dest_file)
    return dest_file

def load_processed_data(dt,d_file):
    delm = get_delimiter(dt)
    cols = get_use_columns(dt)
    print("reading from %s " %d_file)
    df = read_csv(d_file,sep=delm,names=cols,header=None,dtype=object)
    return df
'''
def save_data(dt,df):
    dest_file = get_dest_file(dt)
    df.to_csv(dest_file, sep='\t', header=False,index=False)

'''def load_source_data(dt):
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
'''
def preprocess(dt):
    dest_file = get_dest_file(dt)
    if dest_file.is_file():
        df = load_processed_data(dt,dest_file)
        return df
        
    df = load_source_data(dt)
    print("total number of queries (before preprocessing): %d" % len(df.index))
    df = df[df['query_text'].notna()]   # we get rid of the rows with query_text is na
    df['query_text'] = df['query_text'].apply(unquote)  #we noticed that some queries are url encoded
    df['query_text'] = df['query_text'].str.lower()  #we dont differntiate between lower and upper case
    #nonasc_df = df[~df.query_text.map(str.isascii)]    # get the statistics of non-ascii queries
    #null_cols = df.columns[df.isnull().any()]
    #print(df[null_cols].isnull().sum())
    #sz_series = df.groupby(['session_id']).size()
    #print(df.loc[df.session_id == ma,['session_id','query_text']])
    df['query_text'] = df['query_text'].str.replace('http\S+', ' ',regex=True)  #we get rid of url
    df['query_text'] = df['query_text'].str.replace('[-_:,.]',' ',regex=True)    #to normalize the use of - and _. we replace it with space
    df['query_text'] = df['query_text'].str.replace(r'[^\w\s]','',regex=True)    #replace all non-langauge and space characters
    df = df[df.query_text.map(str.isascii)] # for the time being we deal with only english based queries
    df.query_text = df['query_text'].str.replace('\s+',' ',regex=True)
    #df['query_text'] = df['query_text'].str.strip()
    df = df[df['query_text'].str.strip().astype(bool)]
    df.query_text = df['query_text'].str.replace('^\d[\.\s\d]+$','0',regex=True)  
    #df = df[~df.id.str.isnumeric()]
    df = df[~ df.query_text.str.isnumeric()]
    dd = df[['session_id','query_text']]
    df = df[dd.ne(dd.shift()).any(1)]   # to get rid of the continous repetations
    df = df[df.groupby('session_id',sort=False)['session_id'].transform('size').between(4, 50, inclusive=True)]
    #df['hashed'] = df.query_text.apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    #df = df[df.groupby('session_id')['session_id'].transform('size') > 4]
    #df = df[df.groupby('session_id')['session_id'].transform('size') < 51]
    print("total number of queries (after preprocessing): %d" % len(df.index))
    print("group statistics")
    print(" %s" % df.groupby(['session_id']).size().describe())
    #(a,b) = tuple(dg.size().agg(['idxmax','max']))
    #(a,b) = tuple(dg.size().agg(['idxmin','min']))
    #print(df.loc[df.session_id == a,['session_id','query_text']])
    save_data(dt,df)
    return df

def tokenize(dt,df):
    from tokenizers import Tokenizer
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers import normalizers
    from tokenizers.normalizers import NFD, StripAccents
    from tokenizers.models import WordLevel
    from tokenizers.trainers import WordLevelTrainer
    #from tokenizers.models import WordPiece
    #from tokenizers.trainers import WordPieceTrainer
    #from tokenizers import ByteLevelBPETokenizer
    #from tokenizers.trainers import WordLevelTrainer
    #from tokenizers.models import BPE
    #from tokenizers.trainers import BpeTrainer
    #print(df.head())
    #print(df.query_text.head())
    #print(df.query_text.to_list())
    #exit(0)
    data_source = get_data_source(dt)
    token_file = Path(data_dir,data_source,'tokenizer.json')
    vocab_file = Path(data_dir,data_source,'vocab.txt')
    corpus_file = Path(data_dir,data_source,'corpus.txt')
    if not corpus_file.is_file():
        print("corpus file is generating")
        corpus = df.groupby('session_id', sort=False)['query_text'].apply(' [sep] '.join)
        corpus.to_csv(corpus_file,header=False,index=False)
    else:
        print("corpus file already generated")
    
    if vocab_file.is_file() and corpus_file.is_file():
        print("corpus and token files already generated")
        return 0

    tokenizer = Tokenizer(WordLevel(unk_token="[unk]"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = WordLevelTrainer(vocab_size=30000,min_frequency=3,special_tokens=["[unk]", "[bos]", "[eos]", "[sep]", "[pad]"])
    tokenizer.train_from_iterator(df.query_text.to_list(),trainer)
    tokenizer.model.save('../Data/semanticscholar/tokenizer/wordlevel')
    

    '''tokenizer = Tokenizer(BPE(unk_token="[unk]"))
    pre_tokenizer = WhitespaceSplit()
    trainer = BpeTrainer(min_frequency=2,special_tokens=["[unk]", "[bos]", "[eos]", "[sep]", "[pad]"])
    tokenizer.train_from_iterator(df.query_text.to_list(),trainer)
    tokenizer.model.save('../Data/semanticscholar/tokenizer/')


    tokenizer = Tokenizer(WordLevel(unk_token="[unk]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = WordLevelTrainer(special_tokens=["[unk]", "[bos]", "[eos]", "[sep]"])
    tokenizer.train_from_iterator(df.query_text.to_list(),trainer)
    tokenizer.save(str(token_file))    
    #bert_tokenizer = Tokenizer(WordPiece(unk_token="[unk]"))
    #bert_tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
    #bert_tokenizer.pre_tokenizer = Whitespace()
    #bert_tokenizer.post_processor = TemplateProcessing(
    #    single="[bos] $0 [eos]",
    #    special_tokens=[("[bos]", 1),("[eos]", 2),],)
    #trainer = WordPieceTrainer(vocab_size=25000,min_frequency=3,special_tokens=["[unk]", "[bos]", "[eos]", "[pad]", "[mask]"])
    #print(df.query_text.to_list())
    #bert_tokenizer.train_from_iterator(df.hashed.to_list(),trainer)
    #bert_tokenizer.save(str(token_file))
    #bert_tokenizer.save_model(directory=data_dir,name='tokenizer')
    #df['range_idx'] = range(0, df.shape[0])
    #df['mean_rank_group'] = df.groupby(['session_id'],sort=False)['range_idx'].transform(np.mean)
    #df['separate_column'] = df['range_idx'] < df['mean_rank_group']
    #df = df.groupby(['session_id'], as_index=False, sort=False)['hashed'].agg(' '.join)
    #df = df.groupby('session_id').agg({'query_text':' '.join}).reset_index()
    with open(token_file) as token_f:
        jdata = json.load(token_f)
        with open(vocab_file, "w") as fd:
            for k in jdata['model']['vocab'].keys():
                print(k, file=fd)
            
    #dct = set(' '.join(df.query_text).split())
    #val_cnts = df['query_text'].str.split().explode().value_counts()
    #print(val_cnts[:10])
    #print(val_cnts[-10:])
    #dct = dict(val_cnts)
    #with open(vocab_file, "w") as fd:
    #    for key in dct:
    #        print(key, file=fd)

    '''

def main():
    for dt in dataset:
        df = preprocess(dt)
        tokenize(dt,df)
        
        

if __name__ == '__main__':
    main()
        

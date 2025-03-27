import pandas as pd
import os
from pathlib import Path
import re

DATA_PATH = 'C:\\Users\\rooty\\UWEC\\Research\\CyberBullyingML\\venv\\cyberbullying-ml\\data\\original\\cyberbullying_tweets'

if __name__ == '__main__':

    dir = Path(DATA_PATH)
    
    all_tweets = []
    all_labels = []
    
    for file in dir.iterdir():
        text = file.read_text(encoding='utf-8') 
        tweets = text.split('\n') 
        label = file.stem.lstrip('80')
        all_tweets += tweets
        all_labels += [label for _ in range(len(tweets))]
    

    df = pd.DataFrame(columns=['tweet', 'label'])
    df.tweet = all_tweets
    df.label = all_labels
    
    df.to_csv('48000_cyberbullying_tweets.csv', encoding='utf-8', index=False)

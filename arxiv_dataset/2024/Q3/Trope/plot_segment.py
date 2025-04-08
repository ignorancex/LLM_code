import nltk
import json
import os

nltk.download('gutenberg')
os.makedirs("data/synopses_segment", exist_ok = True)
files = json.load(open("data/test.json"))

for file in files:
    data = json.load(open(f"data/synopses/{file}.json"))
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    text = data['plot']
    sents = sent_tokenizer.tokenize(text)
    data['plot'] = sents

    with open(f'data/synopses_segment/{file}.json', 'w') as fp:
        json.dump(data, fp, indent=4)


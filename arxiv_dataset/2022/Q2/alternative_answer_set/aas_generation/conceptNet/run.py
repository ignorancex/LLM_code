import json
import requests
import argparse
from tqdm.auto import tqdm

def getSynonyms(value):
    hyper = requests.get('http://api.conceptnet.io/query?node=/c/en/'+value+'&rel=/r/IsA')
    syn = requests.get('http://api.conceptnet.io/query?node=/c/en/'+value+'&rel=/r/Synonym')
    label_hyper_json = hyper.json()
    label_syn_json = syn.json()
    # json.dump(label_syn_json, open('./new.json', 'w'))
    label_syn = []

    for data in label_syn_json['edges']:
        if data['end']['language'] == 'en' and not data['end']['label'] in label_syn:
            label_syn.append(data['end']['label'])
        if data['start']['language'] == 'en' and not data['start']['label'] in label_syn:
            label_syn.append(data['start']['label'])

    for data in label_hyper_json['edges']:
        if not data['end']['label'] in label_syn:
            label_syn.append(data['end']['label'])
        if not data['start']['label'] in label_syn:
            label_syn.append(data['start']['label'])

    return label_syn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the conceptNet synset of each label')
    parser.add_argument('--label_file',  type=str, default='data/gqa/trainval_label2ans.json',
                        help='the path of label file of dataset')
    parser.add_argument('--save_file',  type=str, default='conceptNet/gqa/conceptNet_aas.json',
                        help='the path to save the aas of each label')
    
    args = parser.parse_args()
    with open(args.label_file) as f:
        labels = json.load(f)
        f.close()

    ans2syn={} 
    for label in tqdm(labels, total=len(labels)):
        
        label_syn = getSynonyms(label)
        ans2syn[label] = label_syn
        
        
    json.dump(ans2syn, open(args.save_file, 'w'), indent=2)
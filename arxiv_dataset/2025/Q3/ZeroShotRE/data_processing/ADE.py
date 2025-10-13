import sys

sys.path.append(".")

import sys
sys.path.append('.')
from utils import recursive_lowercase, unlist, pickle_save, pickle_load
from universal_classes import Entity, Relation, Example, Dataset
import jsonlines
from collections import defaultdict


# preprocessing with the raw ADE files
def preprocess_ade(file):
    data_raw = list(jsonlines.open(file))
    sentences = dict()
    relations = defaultdict(set)
    gpt_data = []

    for row in data_raw:
        text = row['Sentence']
        text_hash = hash(text)
        sentences[text_hash] = text
        relations[text_hash].add((row['drug'], row['ADE']))

    for key in sentences.keys():
        rel = []
        span = []
        sentence = sentences[key]
        relation = relations[key]
        for entities in relation:
            entity_drug = {'strings': set([entities[0]]), 'entity_type': 'drug'}
            entity_drug = Entity(**entity_drug)
            span.append(entity_drug)

            entity_ade = {'strings': set([entities[1]]), 'entity_type': 'ade'}
            entity_ade = Entity(**entity_ade)
            span.append(entity_ade)

            rel_dict = {'entities': set([entity_drug, entity_ade]), 'relation_type': "ade"}
            rel.append(Relation(**rel_dict))

        if rel:
            example = {'text': sentence, 'relations': set(rel), 'entities': span}
            gpt_data.append(Example(**example))
        else:
            example = {'text': sentence, 'relations': set(), 'entities': span}
            gpt_data.append(Example(**example))

    dataset = {'examples': gpt_data}
    return Dataset(**dataset)


# file names
file_name = 'data/raw/ADE/dataset.jsonl'
data = preprocess_ade(file_name)
# saving directory
output_data_dir = 'data/processed/ADE'
# saving files
pickle_save(data, output_data_dir + 'train_data.save')

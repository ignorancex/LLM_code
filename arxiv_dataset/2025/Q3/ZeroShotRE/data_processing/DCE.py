#%% libraries
import sys
sys.path.append('.')
from utils import pickle_save
from universal_classes import Entity, Relation, Example, Dataset
import jsonlines

# preprocessing with the raw DCE files
def preprocess(file_name):
    data_raw = list(jsonlines.open(file_name))
    rel_map = {'COMB': 'non-positive combination','POS':'positive combination'}
    gpt_data = []
    for example in data_raw:
        # doc_id = example['doc_id']
        spans = []
        sentence = example['sentence']
        spans_dic = example['spans']
        paragraph = example['paragraph']
        for span_dic in spans_dic:
            entity = {'strings': set([span_dic['text']]), 'entity_type': 'drug'}
            entity = Entity(**entity)
            spans.append(entity)

        rels = example['rels']
        if not rels:
            relation = []
        else:
            relation = []
            for rel in rels:
                cls = rel['class']
                if cls == 'NEG':
                    cls = 'COMB'
                curr_spans = []
                for i in rel['spans']:
                    curr_spans.append(spans[i])
                rel_dict = {'entities': set(curr_spans), 'relation_type': rel_map[cls]}
                relation.append(Relation(**rel_dict))
        if relation:
            example = {'title': sentence, 'text': paragraph, 'relations': set(relation), 'entities': spans}
            gpt_data.append(Example(**example))
        else:
            example = {'title': sentence, 'text': paragraph, 'relations': set(), 'entities': spans}
            gpt_data.append(Example(**example))
    dataset = {'examples': gpt_data}
    return Dataset(**dataset)

# file names
train_file_name = 'data/raw/DCE/final_train_set.jsonl'
test_file_name = 'data/raw/DCE/final_test_set.jsonl'

train_data = preprocess(train_file_name)
test_data = preprocess(test_file_name)

# saving directoiry
output_data_dir = 'data/processed/DCE'

# saving files
pickle_save(train_data, output_data_dir + '/' + 'train_data.save')
pickle_save(test_data, output_data_dir + '/' + 'test_data.save')

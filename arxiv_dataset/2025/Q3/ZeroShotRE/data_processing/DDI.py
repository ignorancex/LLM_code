# %% preliminaries
# run as `python -m data_processing.DDI` after navigating to BioIFT directory

# %% libraries
import os
from datasets import load_dataset

from dataclasses import dataclass
from universal_classes import Entity, Relation, Example, Dataset
from utils import make_dir, pickle_save

# %%
@dataclass
class DDIMention:
    
    annotation: ...
    
    def _get_id(self):
        self.id = self.annotation['id']
    
    def _get_string(self):
        self.string = self.annotation['text'].lower()
    
    def _get_type(self):
        self.entity_type = self.annotation['type'].lower()

    def _get_offsets(self):
        self.offsets = tuple(self.annotation['offsets'])

    def return_Entity(self):
        return Entity({self.string}, self.entity_type)

    def __post_init__(self):
        self._get_id()
        self._get_string()
        self._get_type()
        self._get_offsets()
  
# %%
@dataclass
class DDIRelation:
    
    annotation: ...
    parent_example: ...
             
    def _get_head(self):
        mention_id = self.annotation['head']['ref_id']
        self.head = self.parent_example.mention_converter[mention_id]
    
    def _get_tail(self):
        mention_id = self.annotation['tail']['ref_id']
        self.tail = self.parent_example.mention_converter[mention_id]
    
    def _get_predicate(self):
        self.predicate = self.annotation['type'].lower()

    def return_Relation(self):
        head = self.head.return_Entity()
        tail = self.tail.return_Entity()
        return Relation({head, tail}, self.predicate)

    def __post_init__(self):
        self._get_head()
        self._get_tail()
        self._get_predicate()
    
# %% example
@dataclass
class DDIExample:

    annotation: ...
        
    def _get_id(self):
        self.id = self.annotation['document_id']
    
    def _get_text(self):
        self.text = self.annotation['text']
    
    def _get_mentions(self):
        self.mentions = [DDIMention(el) for el in self.annotation['entities']]
    
    def _get_mention_converter(self):
        ids = [el.id for el in self.mentions]
        self.mention_converter = dict(zip(ids, self.mentions))
    
    def _get_relations(self):
        self.relations = [DDIRelation(el, self) for el in self.annotation['relations']]

    def return_Example(self):
        relations = set(el.return_Relation() for el in self.relations)
        return Example(relations, self.text)

    def __post_init__(self):
        self._get_id()
        self._get_text()
        self._get_mentions()
        self._get_mention_converter()
        self._get_relations()
    
    
# %% Dataset
@dataclass
class DDIDataset:
    
    annotation: ...
                
    def _get_examples(self):
        self.examples = [DDIExample(el) for el in self.annotation]
    
    def return_Dataset(self):
        return Dataset([el.return_Example() for el in self.examples])

    def __post_init__(self):
        self._get_examples()
        

# %%

# %%  paths
output_data_path = 'data/processed/DDI'

#%% loading data
huggingface_dataset = load_dataset('bigbio/ddi_corpus')

train_data = huggingface_dataset['train']
test_data = huggingface_dataset['test']

#%% make datasets
train_data = DDIDataset(train_data)
test_data = DDIDataset(test_data)

train_data = train_data.return_Dataset()
test_data = test_data.return_Dataset()

#%% saving datasets
make_dir(output_data_path)

pickle_save(train_data, f'{output_data_path}/train_data.save')    
pickle_save(test_data, f'{output_data_path}/test_data.save')    




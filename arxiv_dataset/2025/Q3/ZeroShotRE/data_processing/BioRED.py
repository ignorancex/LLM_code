#%% libraries
import os
import torch as t

from collections import Counter, defaultdict
from bioc import biocxml, biocjson
from copy import deepcopy
from dataclasses import dataclass

from utils import recursive_lowercase, unlist, pickle_save
from universal_classes import Entity, Relation, Example, Dataset


#%% useful functions
def reverse_dict(input_dict):
    if len(input_dict) > len(set(input_dict.values())):
        return 'dict is not 1-1'
    else:
        new_dict = dict(zip(input_dict.values(), 
                            input_dict.keys()))
        
        return new_dict


#%% mentions
@dataclass
class BioREDMention():
    node: ...
    parent_passage: ...
        
    def _get_type(self):
        
        self.type = self.node.infons['type']
    
    def _get_string(self):
        self.string = self.node.text
        
    def _get_ids(self):
        self.ids = self.node.infons['identifier']
        
        if self.ids == '-':
            self.ids = None
        elif self.type == 'SequenceVariant':
            self.ids = [self.ids]
        else:
            self.ids = self.ids.split(",")
    
    def _no_id(self):
        return self.ids is None
                            
    def __post_init__(self):
        
        self._get_type()
        self._get_string()
        self._get_ids()
    
#%% title and abstract class
@dataclass
class BioREDPassage:
        
    passage: ...
    parent_example: ...
                        
    def _get_text(self):
        self.text = self.passage.text
    
    def _get_mentions(self):
        self.mentions = [BioREDMention(el, self) for el in self.passage.annotations] 
        
    def __post_init__(self):
        self._get_text()
        self._get_mentions()

#%% Entity class
@dataclass
class BioREDDocEntity:
        
    id: ...
    parent_example: ...

    def _get_mentions(self):
        self.mentions = [el for el in self.parent_example.mentions if ((el.ids is not None) and (self.id in el.ids))]

    def _get_type(self):
        self.type = self.mentions[0].type
    
    def _get_strings(self):
        self.strings = set(el.string for el in self.mentions if el.string)
    
    def return_Entity(self):
        return Entity(recursive_lowercase(self.strings), self.type, self.id)
    
    def __post_init__(self):
        self._get_mentions()
        self._get_type()
        self._get_strings()
        

#%% relation class 
@dataclass
class BioREDRelation:
    
    node: ...
    parent_example: ...
        
    def _get_entities(self):
        infons = self.node.infons
        
        self.entity1_id = infons['entity1']
        self.entity2_id = infons['entity2']
        
        self.entity1 = self.parent_example.id2entity[self.entity1_id]
        self.entity2 = self.parent_example.id2entity[self.entity2_id]
    
        
    def _get_predicate(self):
        infons = self.node.infons
        self.predicate = infons['type']
    
    def return_Relation(self):
        entity1 = self.entity1.return_Entity()
        entity2 = self.entity2.return_Entity()
        relation = Relation({entity1, entity2}, self.predicate)
        
        return relation
        
    def __post_init__(self):
        self._get_entities()
        self._get_predicate()
    
#%% example class
@dataclass
class BioREDExample:
    
    node: ...    
    
    def _get_passages(self):
        
        passages = [BioREDPassage(el, self) for el in self.node.passages]
        
        passages[0].passage_type = 'title'
        passages[1].passage_type = 'abstract'
        
        self.passages = passages
    
    def _get_text(self):
        self.title = self.passages[0].text
        self.abstract = self.passages[1].text
    
    def _get_mentions(self):
        self.mentions = self.passages[0].mentions + self.passages[1].mentions
        
    def _get_example_entities(self):
        ids = set()
        for el in self.mentions:
            if el.ids is not None:
                ids.update(set(el.ids))
        
        self.entities = [BioREDDocEntity(el, self) for el in ids]
        self.id2entity = dict(zip(ids, self.entities))
        
    def _get_relations(self):
        self.relations = [BioREDRelation(el, self) for el in self.node.relations]
     
    def return_Example(self):
        
        relations = {el.return_Relation() for el in self.relations}
        text = self.abstract
        title = self.title
        entities = {el.return_Entity() for el in self.entities}
        
        return Example(relations, text, title, entities)    
     
    def __post_init__(self):
        self._get_passages()
        self._get_text()
        self._get_mentions()
        self._get_example_entities()
        self._get_relations()
        
    

#%%
@dataclass
class BioREDDataset: 

    documents: ...
        
    def _get_examples(self):
        self.examples = [BioREDExample(el) for el in self.documents]
        
    def __post_init__(self):
        self._get_examples()    
    
    def __iter__(self, i):
        return self.examples[i]
    
    def return_Dataset(self):
        return Dataset([el.return_Example() for el in self.examples])
    
    @classmethod
    def from_collection(cls, collection):
        return BioREDDataset(collection.documents)    
    
#%% settings
raw_data_dir = 'data/raw/BioRED'
output_data_dir = 'data/processed/BioRED'

#%% loading data
train_file = raw_data_dir + '/' + 'Train.BioC.XML'
with open(train_file, 'r') as fp:
    train_collection = biocxml.load(fp)
    
valid_file = raw_data_dir + '/' + 'Dev.BioC.XML'
with open(valid_file, 'r') as fp:
    valid_collection = biocxml.load(fp)
    
test_file = raw_data_dir + '/' + 'Test.BioC.XML'
with open(test_file, 'r') as fp:
    test_collection = biocxml.load(fp)

#%% 
train_data = BioREDDataset.from_collection(train_collection)
valid_data = BioREDDataset.from_collection(valid_collection)
test_data = BioREDDataset.from_collection(test_collection)

train_data = train_data.return_Dataset()
valid_data = valid_data.return_Dataset()
test_data = test_data.return_Dataset()

pickle_save(train_data, output_data_dir + '/' + 'train_data.save')    
pickle_save(valid_data, output_data_dir + '/' + 'valid_data.save')    
pickle_save(test_data, output_data_dir + '/' + 'test_data.save')  


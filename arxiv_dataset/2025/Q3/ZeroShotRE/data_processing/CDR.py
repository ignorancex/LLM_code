#%% libraries
import os
import torch as t

from collections import Counter, defaultdict
from bioc import biocxml, biocjson
from copy import deepcopy
from dataclasses import dataclass

from utils import recursive_lowercase, unlist, pickle_save
from universal_classes import Entity, Relation, Example, Dataset

#%% mentions
@dataclass
class CDRMention:
    
    annotation: ...
    parent_passage: ...
    
    def _get_string(self):
        self.string = self.annotation.text

    def _get_ids(self):
        self.ids = self.annotation.infons['MESH']
        
        if self.ids == '-':
            self.ids = None
        else:
            self.ids = self.ids.split("|")
            self.is_composite = True
    
    def _get_type(self):
        self.type = self.annotation.infons['type'].lower()        
    
    def _no_id(self):
        return self.ids is None
                            
    def __post_init__(self):
        self.parent_document = self.parent_passage.parent_example.document
        self._get_string()
        self._get_ids()
        self._get_type()
        
    
#%% title and abstract class
@dataclass
class CDRPassage:
        
    passage: ...
    parent_example: ...
                        
    def _get_text(self):
        self.text = self.passage.text
    
    def _get_mentions(self):
        self.mentions = [CDRMention(el, self) for el in self.passage.annotations] 
        
    def __post_init__(self):
        self._get_mentions()
        self._get_text()
        
#%% Entity class
@dataclass
class CDRDocEntity:
        
    id: ...
    parent_example: ...
            
    def _get_mentions(self):
        
        self.mentions = [el for el in self.parent_example.mentions if self.id in el.ids]
    
    def _get_type(self):
        
        self.type = self.mentions[0].type
    
    def _get_strings(self):
        strings = [el.string for el in self.mentions if el.string]
        self.strings = set(strings)
        
    def return_Entity(self):
        return Entity(recursive_lowercase(self.strings), self.type, self.id)
   
    def __post_init__(self):
        self._get_mentions()
        self._get_type()
        self._get_strings()
        
#%% relation class 
@dataclass
class CDRRelation:
        
    node: ...
    parent_example: ...
        
    def _get_entities(self):
        infons = self.node.infons
        
        self.head_id = infons['Chemical']
        self.tail_id = infons['Disease']
        
        self.head = self.parent_example.id2entity[self.head_id]
        self.tail = self.parent_example.id2entity[self.tail_id]
        
    def return_Relation(self):
        
# =============================================================================
#         chemical = Entity(recursive_lowercase(self.head.strings), 'chemical', self.head_id)
#         disease = Entity(recursive_lowercase(self.tail.strings), 'disease', self.tail_id)
# =============================================================================
        
        chemical = self.head.return_Entity()
        disease = self.tail.return_Entity()
        
        relation = Relation({chemical, disease}, '')
        
        return relation
    
    def __post_init__(self):
        self._get_entities()

    
    
#%% example class
@dataclass
class CDRExample:
    
    document: ...
    
    def _get_passages(self):
        
        passages = [CDRPassage(el, self) for el in self.document.passages]
        
        passages[0].passage_type = 'title'
        passages[1].passage_type = 'abstract'
        
        self.passages = passages
    
    def _get_text(self):
        self.title_text = self.passages[0].text
        self.abstract_text = self.passages[1].text
        
    def _get_mentions(self):
        self.mentions = self.passages[0].mentions + self.passages[1].mentions
    
    def _get_example_entities(self):
        unique_ids = set(unlist([el.ids for el in self.mentions if el.ids is not None]))    
        self.entities = [CDRDocEntity(el, self) for el in unique_ids]
        self.id2entity = dict(zip(unique_ids, self.entities))
                
    def _get_relations(self):
        self.relations = [CDRRelation(el, self) for el in self.document.relations]
    
    def return_Example(self):
        
        relations = {el.return_Relation() for el in self.relations}
        text = self.abstract_text
        title = self.title_text
        entities = {el.return_Entity() for el in self.entities}
        
        return Example(relations, text, title, entities)
        
    
    ############ processing
    def __post_init__(self):
        self._get_passages()
        self._get_text()
        self._get_mentions()
        self._get_example_entities()
        self._get_relations()
    

    
#%% dataset class
@dataclass
class CDRDataset:

    documents: ...
        
    def _get_examples(self):
        self.examples = [CDRExample(el) for el in self.documents]
        
    def __post_init__(self):
        self._get_examples()    
    
    def __iter__(self, i):
        return self.examples[i]
    
    def return_Dataset(self):
        return Dataset([el.return_Example() for el in self.examples])
    
    @classmethod
    def from_collection(cls, collection):
        return CDRDataset(collection.documents)
    

#%% settings
raw_data_dir = 'data/raw/CDR/CDR.Corpus.v010516'
output_data_dir = 'data/processed/CDR'

#%% loading data
train_file = raw_data_dir + '/' + 'CDR_TrainingSet.BioC.xml'
with open(train_file, 'r') as fp:
    train_collection = biocxml.load(fp)
    
valid_file = raw_data_dir + '/' + 'CDR_DevelopmentSet.BioC.xml'
with open(valid_file, 'r') as fp:
    valid_collection = biocxml.load(fp)
    
test_file = raw_data_dir + '/' + 'CDR_TestSet.BioC.xml'
with open(test_file, 'r') as fp:
    test_collection = biocxml.load(fp)

#%% 
train_data = CDRDataset.from_collection(train_collection)
valid_data = CDRDataset.from_collection(valid_collection)
test_data = CDRDataset.from_collection(test_collection)

train_data = train_data.return_Dataset()
valid_data = valid_data.return_Dataset()
test_data = test_data.return_Dataset()

pickle_save(train_data, output_data_dir + '/' + 'train_data.save')    
pickle_save(valid_data, output_data_dir + '/' + 'valid_data.save')    
pickle_save(test_data, output_data_dir + '/' + 'test_data.save')    



# %% libraries
from typing import Optional, Set, List
from dataclasses import dataclass, field, replace
from collections import defaultdict
from copy import deepcopy
import random
from random import sample, shuffle
from utils import recursive_lowercase


# %% classes
@dataclass(frozen=True)
class Entity:
    strings: Set[str]
    entity_type: str
    entity_id: Optional[str] = field(default=None)

    def __hash__(self):
        entity_type = self.entity_type
        if self.entity_id:
            return hash((self.entity_id, entity_type))
            # return hash(self.entity_id)
        else:
            string = recursive_lowercase(list(self.strings)[0])
            return hash((string, entity_type))

    def __eq__(self, other):
        is_same_entity_type = self.entity_type == other.entity_type
        is_string_subset = self.strings.issubset(other.strings) or other.strings.issubset(self.strings)
        is_same_id = self.entity_id == other.entity_id and self.entity_id is not None

        return is_same_entity_type and (is_string_subset or is_same_id)

    def add_entity_id(self, _id):
        if isinstance(_id, set):
            _id = list(_id)[0]
        return replace(self, entity_id = _id)

@dataclass(frozen=True)
class Relation:
    entities: Set[Entity]
    relation_type: str

    def __hash__(self):
        return hash((frozenset(self.entities), recursive_lowercase(self.relation_type)))

    def __eq__(self, other):
        return self.relation_type == other.relation_type and self.entities == other.entities

    def add_entity_ids(self, entities_with_ids):
        return replace(self, entities=entities_with_ids)

    def duplicate(self):
        for el_entity in self.entities:
            [deepcopy(self).replace()]

        return [deepcopy(self).replace(my_set={el}) for el in self.my_set]

def recursive_lowercase2(data):
    if isinstance(data, str):
        return data.lower()
    elif isinstance(data, list):
        return [recursive_lowercase2(el) for el in data]
    elif isinstance(data, set):
        return {recursive_lowercase2(el) for el in data}
    elif isinstance(data, dict):
        return {key: recursive_lowercase2(value) for key, value in data.items()}
    elif isinstance(data, Entity):
        return Entity(recursive_lowercase2(data.strings), recursive_lowercase2(data.entity_type), recursive_lowercase2(data.entity_id))
    elif isinstance(data, Relation):
        return Relation(recursive_lowercase2(data.entities), recursive_lowercase2(data.relation_type))
    else:
        return data

@dataclass
class Example:
    relations: Set[Relation]
    text: str
    title: Optional[str] = field(default = None)
    entities: Optional[Set[Entity]] = field(default = None)


@dataclass
class Dataset:
    examples: List[Example]

    def __getitem__(self, i):
        return self.examples[i]

    def __iter__(self):
        return iter(self.examples)

    def __len__(self):
        return len(self.examples)

    def random_subset(self, sample_size, seed):
        random.seed(seed)
        shuffle(self.examples)
        self.original_examples = deepcopy(self.examples)
        self.examples = self.examples[:sample_size]


@dataclass
class Oracle:
    entities: Set[Entity]

    def __post_init__(self):
        self._get_string2id()

    def _get_string2id(self):

        self.string2id = defaultdict(set)

        def workhorse(entity):
            for el in entity.strings:
                ID = entity.entity_id
                string = el.lower()

                self.string2id[el.lower()].add(entity.entity_id)

        for el in self.entities:
            workhorse(el)

    def convert_string(self, string):
        return self.string2id[recursive_lowercase(string)]

    def convert_relation(self, relation):
        
        def get_ids(relation):
            new_entities = set()
            for el in relation.entities:
                string = list(el.strings)[0]
                ids = self.convert_string(string)
                if len(ids) == 0:
                    ids = None
                elif len(ids) == 1:
                    ids = list(ids)[0]
                normalized_entity = el.add_entity_id(ids)
                new_entities.add(normalized_entity)
                
            relation = replace(relation, entities = new_entities)

            return relation

        def is_composite_entity(entity):
            _id = entity.entity_id
            if isinstance(_id, set) and len(_id) > 1:
                return True
            else:
                return False

        def relation_is_complete(relation):
            for el in relation.entities:
                if is_composite_entity(el):
                    return False
            return True

        def explode(relation):

            def explode_inner(composite_entity, relation):

                other_entities = relation.entities - {composite_entity}


                def workhorse(_id):
                    relation_copy = deepcopy(relation)
                    composite_entity_copy = deepcopy(composite_entity)
                    
                    new_entity = replace(composite_entity_copy, entity_id = _id)
                    new_entities = other_entities | {new_entity}
                    relation_copy = replace(relation_copy, entities=new_entities)

                    return relation_copy

                return {workhorse(el) for el in composite_entity.entity_id}
            
            
            exploded_relations = set()
            
            for el in relation.entities:
                if is_composite_entity(el):
                    new_relations = explode_inner(el, relation)
                    exploded_relations.update(new_relations)
            
                return exploded_relations

        
        relation = get_ids(relation)
        
        complete_relations = set()
        incomplete_relations = {relation}

        while len(incomplete_relations) > 0:
            relation = incomplete_relations.pop()
            if relation_is_complete(relation):
                complete_relations.add(relation)
            else:
                incomplete_relations |= explode(relation)

        return complete_relations

    def convert_relations(self, relations):
        # print('relations')
        converted_relations = set()
        # print('type: ', type(relations))
        for el in relations:
            converted_relations |= self.convert_relation(el)

        return converted_relations

    def __call__(self, relations):
        return self.convert_relations(relations)

@dataclass
class ExampleScorer:

    '''Base class for our scorers'''

    gold_relations: Set[Relation]
    predicted_relations: Set[Relation]

    def __post_init__(self):
        self.score_relations()

    def score_relations(self):
        self.TP_relations = self.gold_relations & self.predicted_relations
        self.FP_relations = self.predicted_relations - self.gold_relations
        self.FN_relations = self.gold_relations - self.predicted_relations

        self.TP = len(self.TP_relations)
        self.FP = len(self.FP_relations)
        self.FN = len(self.FN_relations)

@dataclass
class LowercaseScorer(ExampleScorer):

    def __post_init__(self):
        self.lowercase_relations()
        self.score_relations()
        
    def lowercase_relations(self):

        self.gold_relations = recursive_lowercase2(self.gold_relations)
        self.predicted_relations = recursive_lowercase2(self.predicted_relations)

@dataclass
class NormalizedScorer(LowercaseScorer):

    def __post__init(self):
        self.lowercase_relations()
        self.filter_failed_normalization()
        self.score_relations()

    def filter_failed_normalization(self):
        def is_normalized(relation):
            for el in relation.entities:
                if el.entity_id is None:
                    return False
            else:
                return True
            
        self.predicted_relations = set(el for el in self.predicted_relations if is_normalized(el)) 


# @dataclass
# class DCEPositiveCombinationExampleScorer:
#     gold_relations: Set[Relation]
#     predicted_relations: Set[Relation]
#
#     def __post_init__(self):
#         self.score_relations()
#
#     def score_relations(self):
#         removed_non_positive_prediction = set()
#         for r in self.predicted_relations:
#             if r.relation_type == 'positive combination':
#                 removed_non_positive_prediction.add(r)
#
#         removed_non_positive_gold = set()
#         for r in self.gold_relations:
#             if r.relation_type == 'positive combination':
#                 removed_non_positive_gold.add(r)
#
#         self.TP_relations = removed_non_positive_gold & removed_non_positive_prediction
#         self.FP_relations = removed_non_positive_prediction - removed_non_positive_gold
#         self.FN_relations = removed_non_positive_gold - removed_non_positive_prediction
#
#         self.TP = len(self.TP_relations)
#         self.FP = len(self.FP_relations)
#         self.FN = len(self.FN_relations)
#
#
# @dataclass
# class DCEAnyCombinationExampleScorer:
#     gold_relations: Set[Relation]
#     predicted_relations: Set[Relation]
#
#     def __post_init__(self):
#         self.score_relations()
#
#     def score_relations(self):
#         unified_relation_type_prediction = set()
#         for r in self.predicted_relations:
#             unified_relation_type_prediction.add(Relation(r.entities, ''))
#
#         unified_relation_type_gold = set()
#         for r in self.gold_relations:
#             unified_relation_type_gold.add(Relation(r.entities, ''))
#
#         self.TP_relations = unified_relation_type_gold & unified_relation_type_prediction
#         self.FP_relations = unified_relation_type_prediction - unified_relation_type_gold
#         self.FN_relations = unified_relation_type_gold - unified_relation_type_prediction
#
#         self.TP = len(self.TP_relations)
#         self.FP = len(self.FP_relations)
#         self.FN = len(self.FN_relations)

@dataclass
class DCEPositiveCombinationExampleScorer(LowercaseScorer):
    gold_relations: Set[Relation]
    predicted_relations: Set[Relation]

    def __post_init__(self):
        self.lowercase_relations()
        self.filter_non_positive_relations()
        self.score_relations()

    def filter_non_positive_relations(self):
        def workhorse(relations):
            return set(el for el in relations if el.relation_type == 'positive combination')
        
        self.gold_relations = workhorse(self.gold_relations)
        self.predicted_relations = workhorse(self.predicted_relations)

@dataclass
class DCEAnyCombinationExampleScorer(LowercaseScorer):

    def __post_init__(self):
        self.lowercase_relations()
        self.combine_relation_types()
        self.score_relations()

    def combine_relation_types(self):
        def workhorse(relations):
            return set(Relation(el.entities, '') for el in relations)
        
        self.gold_relations = workhorse(self.gold_relations)
        self.predicted_relations = workhorse(self.predicted_relations)


class F1Calculator:

    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def update(self, TP, FP, FN):
        self.TP += TP
        self.FP += FP
        self.FN += FN

    def _compute_precision(self):
        if self.TP + self.FP == 0:
            return 0
        else:
            return self.TP / (self.TP + self.FP)

    def _compute_recall(self):
        if self.TP + self.FN == 0:
            return 0
        else:
            return self.TP / (self.TP + self.FN)

    def _compute_F1(self):
        if self.TP + self.FP + self.FN == 0:
            return 0
        else:
            return 2 * self.TP / (2 * self.TP + self.FP + self.FN)

    def compute(self):
        output = dict(precision=self._compute_precision(),
                      recall=self._compute_recall(),
                      F1=self._compute_F1()
                      )
        return output

# =============================================================================
# #%% Entity unit testing
# ### two exactly equal entities are equal
# # entity-level datasets
# ent1 = Entity({'a', 'b'}, 'et1', '1')  
# ent2 = Entity({'a', 'b'}, 'et1', '1')  
# 
# ent1 == ent2
# 
# # mention-level datasets
# ent1 = Entity({'a'}, 'et1')  
# ent2 = Entity({'a'}, 'et1')  
# 
# ent1 == ent2
# 
# ### two entities are equal if ids are equal/one set of entities is a subset of the other
# # entity-level datasets
# ent1 = Entity({'a', 'b'}, 'et1', '1')  
# ent2 = Entity({'b'}, 'et1', '1')  
# 
# ent1 == ent2
# 
# # mention-level datsets, they'll be exactly the same, so trivial
# 
# ### two entities are unequal if strings are not subsets (and ids don't match, for entity-level datasets)
# # entity-level
# ent1 = Entity({'a', 'b'}, 'et1', '1')  
# ent2 = Entity({'c'}, 'et1', '2')  
# 
# ent1 == ent2
# 
# # mention-level
# ent1 = Entity({'a'}, 'et1')  
# ent2 = Entity({'b'}, 'et1')  
# 
# ent1 == ent2
# 
# # entity-level scenario that should never happen, unless dataset errors or preprocessing errors
# ent1 = Entity({'a', 'b'}, 'et1', '1')  
# ent2 = Entity({'c'}, 'et1', '1')  
# 
# ent1 == ent2
# 
# ### two entities are unequal if entity types are unequal
# # entity-level
# ent1 = Entity({'a', 'b'}, 'et1', '1')  
# ent2 = Entity({'a', 'b'}, 'et2', '1')  
# 
# ent1 == ent2
# 
# # mention-level
# ent1 = Entity({'a'}, 'et1')  
# ent2 = Entity({'a'}, 'et2')  
# 
# ent1 == ent2
# 
# 
# 
# #%% Relation unit testing
# ### two identical gold relations are equal
# ent1 = Entity({'a', 'b'}, 'et1', '1')  
# ent2 = Entity({'c', 'd'}, 'et2', '2') 
# 
# rel1 = Relation({ent1, ent2}, 'rt1')
# rel2 = Relation({ent1, ent2}, 'rt1')
# 
# rel1 == rel2
# 
# ### a set of gold_relations contains no duplicates
# len({rel1, rel2})
# 
# ### a predicted relation equals a gold relation if the entities are equal and relation type is equal
# gold_ent1 = Entity({'a', 'b'}, 'et1', '1')  
# gold_ent2 = Entity({'c', 'd'}, 'et2', '2') 
# 
# gold_rel = Relation({gold_ent1, gold_ent2}, 'rt1')
# 
# pred_ent1 = Entity({'a'}, 'et1', '1')  
# pred_ent2 = Entity({'c'}, 'et2', '2') 
# 
# pred_rel = Relation({pred_ent1, pred_ent2}, 'rt1')
# 
# pred_rel == gold_rel
# 
# ### a predicted relation does not equal a gold relation if the entities are equal but relation type is unequal
# gold_ent1 = Entity({'a', 'b'}, 'et1', '1')  
# gold_ent2 = Entity({'c', 'd'}, 'et2', '2') 
# 
# gold_rel = Relation({gold_ent1, gold_ent2}, 'rt1')
# 
# pred_ent1 = Entity({'a'}, 'et1', '1')  
# pred_ent2 = Entity({'c'}, 'et2', '2') 
# 
# pred_rel = Relation({pred_ent1, pred_ent2}, 'rt2')
# 
# pred_rel == gold_rel
# 
# ### a predicted relation does not equal a gold relation if entities are unequal
# gold_ent1 = Entity({'a', 'b'}, 'et1', '1')  
# gold_ent2 = Entity({'c', 'd'}, 'et2', '2') 
# 
# gold_rel = Relation({gold_ent1, gold_ent2}, 'rt1')
# 
# pred_ent1 = Entity({'a'}, 'et1', '1')  
# pred_ent2 = Entity({'z'}, 'et2', '3') 
# 
# pred_rel = Relation({pred_ent1, pred_ent2}, 'rt1')
# 
# pred_rel == gold_rel
# 
# ### duplication of predicted due to composite mentions
# gold_ent1 = Entity({'a'}, 'et1', '1')  
# gold_ent2 = Entity({'a'}, 'et1', '2')  
# gold_ent3 = Entity({'b'}, 'et2', '3')  
# gold_ent4 = Entity({'b'}, 'et2', '4')
# gold_ents = {gold_ent1, gold_ent2, gold_ent3, gold_ent4}
# 
# oracle = Oracle(gold_ents)
# 
# pred_ent1 = Entity({'a'}, 'et1')
# pred_ent2 = Entity({'b'}, 'et2')
# pred_rel = Relation({pred_ent1, pred_ent2}, '')
# pred_rels = {pred_rel}
# pred_rels = oracle(pred_rels)
# 
# pred_rels
# 
# #%% Oracle unit testing
# 
# ### string-to-id converter is correct
# gold_ent1 = Entity({'a', 'b'}, 'et1', '1')  
# gold_ent2 = Entity({'c', 'd'}, 'et2', '2')
# gold_ent3 = Entity({'e', 'f'}, 'et1', '3')
# 
# gold_ents = {gold_ent1, gold_ent2, gold_ent3}
# 
# oracle = Oracle(gold_ents)
# 
# oracle.string2id
# 
# ### converts predicted relations correctly
# pred_ent1 = Entity({'a'}, 'et3')  
# pred_ent2 = Entity({'c'}, 'et4') 
# 
# pred_rel = Relation({pred_ent1, pred_ent2}, 'rt2')
# 
# pred_rels = {pred_rel}
# 
# oracle(pred_rels)
# 
# ### handles composite mentions correctly
# gold_ent1 = Entity({'a'}, 'et1', '1')  
# gold_ent2 = Entity({'b'}, 'et2', '1')
# 
# #%% ExampleScorer unit testing
# 
# gold_ent1 = Entity({'a', 'b'}, 'et1', '1')  
# gold_ent2 = Entity({'c', 'd'}, 'et2', '2') 
# gold_ent3 = Entity({'e', 'f'}, 'et1', '3')  
# 
# gold_rel1 = Relation({gold_ent1, gold_ent2}, 'rt1')
# gold_rel2 = Relation({gold_ent3, gold_ent2}, 'rt1')
# 
# gold_rels = {gold_rel1, gold_rel2}
# oracle = Oracle.from_gold_relations(gold_rels) 
# 
# 
# pred_ent1 = Entity({'a'}, 'et1')  
# pred_ent2 = Entity({'c'}, 'et2') 
# pred_ent3 = Entity({'h'}, 'et1')
# 
# pred_rel1 = Relation({pred_ent1, pred_ent2}, 'rt1')
# pred_rel2 = Relation({pred_ent3, pred_ent2}, 'rt1')
# 
# pred_rels = {pred_rel1, pred_rel2}
# pred_rels = oracle(pred_rels)
# 
# scorer = ExampleScorer(gold_rels, pred_rels)
# 
# 
# scorer.TP
# scorer.FP
# scorer.FN
# 
# scorer.TP_relations
# scorer.FP_relations
# scorer.FN_relations        
# 
# =============================================================================

# %%

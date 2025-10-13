from universal_classes import Entity, Relation  
from copy import deepcopy


path_system_prompt = 'prompts/BioRED.txt'
with open(path_system_prompt, 'r') as file:
    system_prompt = file.read()

#%%
class BioREDTemplate_schema:

    text = {
        "type": "string",
        "description": "Text of entity"
        }
    
    entity_type = {
        "type": "string",
        "description": "One of the following categories: GeneOrGeneProduct, SequenceVariant, ChemicalEntity, or DiseaseOrPhenotypicFeature"
        }
    
    entity1 = {"type": "object",
               "description": "A biomedical entity that is either a GeneOrGeneProduct, SequenceVariant, ChemicalEntity, or DiseaseOrPhenotypicFeature",
               "properties": {
                   "text": text,
                   "entity_type": entity_type
                   },
               "required": ["text", "entity_type"]
        }
    
    entity2 = deepcopy(entity1)
    
    relation = {
        "type": "object",
        "description": "Two biomedical entities and a relationship that holds between them.",
        "properties": {
            "entity1": entity1,
            "entity2": entity2,
            "relation": {
            "type": "string",
            "description": "A biomedical relationship that holds between two entities, either Positive_Correlation, Negative_Correlation, Association, Bind, Cotreatment, Drug_Interaction, Comparison, or Conversion"
        }
            },
        "required": ["entity1", "entity2", "relation"]
        }
    
    relations = {
        "type": "array",
        "description": "List of biomedical relations",
        "items": relation 
        }
    
    top_level_object = {"relations": relations}
    
    description = 'Extracts a list of relations from a text.'
    
    schema = {
        "name": "extract_relations",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": top_level_object
            },
        "required": ["relations"]
        }
    

    @classmethod
    def extract_relations(cls, relation_list):
        relations = set()
        for el in relation_list:
            entity1 = Entity({el['entity1']['text']}, el['entity1']['entity_type'])
            entity2 = Entity({el['entity2']['text']}, el['entity2']['entity_type'])
            relation = Relation({entity1, entity2}, el['relation'])
            relations.add(relation)

        return relations

    @classmethod
    def make_prompt(cls, example):
        
        system_content = system_prompt
        
        user_content = f'Title: {example.title}\n\nAbstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]
        
        return messages
    


class BioREDTemplate_json:

    @classmethod
    def extract_relations(cls, relation_list):
        relations = set()
        for el in relation_list:
            entity1 = Entity({el['entity1']['text']}, el['entity1']['entity_type'])
            entity2 = Entity({el['entity2']['text']}, el['entity2']['entity_type'])
            relation = Relation({entity1, entity2}, el['relation'])
            relations.add(relation)

        return relations

    @classmethod
    def make_prompt(cls, example):
        system_content = system_prompt
        user_content = f'Title: {example.title}\n\nAbstract: {example.text}'
        system = {'role': 'system',
                  'content': "You are a helpful assistant designed to output JSON." + system_content}
        user = {'role': 'user',
                'content': user_content}
        messages = [system, user]

        return messages
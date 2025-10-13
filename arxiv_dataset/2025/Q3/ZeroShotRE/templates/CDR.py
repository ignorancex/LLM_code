#%% libraries
from universal_classes import Entity, Relation 


path_system_prompt = 'prompts/CDR.txt'
with open(path_system_prompt, 'r') as file:
    system_prompt = file.read()

#%% Templates
class CDRTemplate_schema:
        
    chemical = {
            "type": "string",
            "description": "Chemical or drug."
        }
    disease = {
            "type": "string",
            "description": "Disease."
        }
    
    items = {
        "type": "object",
        "properties": {
            "chemical": chemical,
            "disease": disease
            },
        "description": "A chemical-disease pair."
        }
    
    relations = {
        "type": "array",
        "items": items,
        "required": ["chemical", "disease"],
        "description": "A list of chemical-induced disease relations."
        }
    
    parameters = {
            "type": "object",
            "properties": {
                "relations": relations
            }
        }

    schema = {
        "name": "extract_relations",
        "description": "Extracts a list of chemical-induced disease relations from a text.",
        "parameters": parameters,
        "required": ["relations"]
    }

    @classmethod
    def extract_relations(cls, relation_list):
        relations = set()
        for el in relation_list:
            chemical = Entity({el['chemical']}, 'chemical')
            disease = Entity({el['disease']}, 'disease')
            relation = Relation({chemical, disease}, '')
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

class CDRTemplate_json:

    @classmethod
    def extract_relations(cls, relation_list):
        relations = set()
        for el in relation_list:
            chemical = Entity({el['chemical']}, 'chemical')
            disease = Entity({el['disease']}, 'disease')
            relation = Relation({chemical, disease}, '')
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

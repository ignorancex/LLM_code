from universal_classes import Entity, Relation
from copy import deepcopy
from utils import recursive_lowercase


path_system_prompt = 'prompts/GDA.txt'
with open(path_system_prompt, 'r') as file:
    system_prompt = file.read()

# %%
class GDATemplate_json:

    @classmethod
    def extract_relations(cls, relation_list):
        relations = set()
        relation_list = recursive_lowercase(relation_list)
        for el in relation_list:
            chemical = Entity({el['gene']}, 'gene')
            gene = Entity({el['disease']}, 'disease')
            relation = Relation({chemical, gene}, 'association')
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

    
class GDATemplate_schema:

    relation = {
        "type": "object",
        "description": "gene-disease associations",
        "properties": {
            "gene": {
                "type": "string",
                "description": "gene"
            },
            "disease": {
                "type": "string",
                "description": "disease"
            },
            "relation": {
            "type": "string",
            "description": "Relations can only be 'association'."
        }
        },
        "required": ["gene", "disease", "relation"]
    }

    relations = {
        "type": "array",
        "description": "List of gene-disease associations",
        "items": relation
    }

    top_level_object = {"relations": relations}

    description = 'Extract relevant gene-disease associations from a text.'

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
        relation_list = recursive_lowercase(relation_list)
        for el in relation_list:
            gene = Entity({el['gene']}, 'gene')
            disease = Entity({el['disease']}, 'disease')
            relation = Relation({gene, disease}, 'association')
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

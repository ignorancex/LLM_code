from universal_classes import Entity, Relation
from copy import deepcopy
from utils import recursive_lowercase

path_system_prompt = 'prompts/ChemProt.txt'
with open(path_system_prompt, 'r') as file:
    system_prompt = file.read()


# %%
class ChemProtTemplate_json:

    @classmethod
    def extract_relations(cls, relation_list):
        relations = set()
        relation_list = recursive_lowercase(relation_list)
        for el in relation_list:
            chemical = Entity({el['chemical']}, 'chemical')
            gene = Entity({el['gene']}, 'gene')
            relation = Relation({chemical, gene}, el['relation'])
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


class ChemProtTemplate_schema:

    relation = {
        "type": "object",
        "description": "Two biomedical entities and a relationship that holds between them.",
        "properties": {
            "chemical": {
                "type": "string",
                "description": "chemical"
            },
            "gene": {
                "type": "string",
                "description": "gene/protein"
            },
            "relation": {
        "type": "string",
        "description": "One of the following 5 predicates: 'CPR:3', 'CPR:4', 'CPR:5', 'CPR:6' and 'CPR:9'."
        }   
        },
        "required": ["chemical", "gene", "relation"]
    }

    relations = {
        "type": "array",
        "description": "List of chemical-protein relations",
        "items": relation
    }

    top_level_object = {"relations": relations}

    description = 'Extract relevant chemical-protein relations from a text.'

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
            chemical = Entity({el['chemical']}, 'chemical')
            gene = Entity({el['gene']}, 'gene')
            relation = Relation({chemical, gene}, el['relation'])
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


#%% libraries
from dataclasses import dataclass
import ast
from universal_classes import Entity, Relation
path_system_prompt = 'prompts/DCE.txt'
with open(path_system_prompt, 'r') as file:
    system_prompt = file.read()

# %%
class DCETemplate_schema:
    function_name = "extract"
    instructions = "Extracts a list of drug combination relations from a text."

    properties = {
        "drugs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "drug": {
                        "type": "string",
                        "description": "drug"
                    }
                }
            }
        },
        "relation": {
            "type": "string",
            "description": "\'positive combination\': the sentence indicates that certain drugs are used in "
                           "combination, and the passage suggests that the combination has additive, synergistic, "
                           "or otherwise beneficial effects which warrant further study.\'non-positive combination\': "
                           "the sentence indicates the drugs are used in combination, but there is no evidence in the "
                           "passage that the effect is positive (it is either negative or undetermined).If the "
                           "sentence does not state that the given drugs are used in combination, even if a "
                           "combination is indicated somewhere else in the wider context, do not identify anything. "
        }
    }

    required = ["drugs", "relation"]

    schema = {
        "name": function_name,
        "description": instructions,
        "parameters": {
            "type": "object",
            "properties": {
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": properties
                    },
                    "required": required
                }
            }
        },
        "required": ["relations"]
    }

    @classmethod
    def extract_relations(cls, relation_list):
        extracted = set()
        for rel in relation_list:
            entities = []
            for e in rel['drugs']:
                entities.append(Entity({e['drug']}, 'drug'))
            entities = set(entities)
            rel = Relation(entities, rel['relation'])
            extracted.add(rel)

        return extracted

    @classmethod
    def make_prompt(cls, example):
        system_content = system_prompt

        user_content = f'Now extract all drug combinations from from the following sentence. Sentence: {example.title}\n\n Passage: {example.text} '

        system = {'role': 'system',
                  'content': "You are a helpful assistant designed to output JSON. " + system_content}

        user = {'role': 'user',
                'content': user_content}

        messages = [system, user]

        return messages


class DCETemplate_json:
    
    @classmethod
    def extract_relations(cls, relation_list):
        extracted = set()
        for rel in relation_list:
            entities = []
            for e in rel['drugs']:
                entities.append(Entity({e}, 'drug'))
            entities = set(entities)
            rel = Relation(entities, rel['relation'])
            extracted.add(rel)

        return extracted

    @classmethod
    def make_prompt(cls, example):
        system_content = system_prompt

        user_content = f'Now extract all drug combinations from from the following sentence. Sentence: {example.title}\n\n Passage: {example.text} '

        system = {'role': 'system',
                  'content': "You are a helpful assistant designed to output JSON. " + system_content}

        user = {'role': 'user',
                'content': user_content}

        messages = [system, user]

        return messages

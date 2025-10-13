# templates
from universal_classes import Entity, Relation
from copy import deepcopy
from utils import recursive_lowercase

path_system_prompt = 'prompts/ADE.txt'
with open(path_system_prompt, 'r') as file:
    system_prompt = file.read()


class AdeTemplate:
    """
    A template has three components:
    (1) `schema`, an attribute that contains the general instructions for GPT-4.
    (2) `make_prompt`, a classmethod that converts an Example into text for GPT-4.
    (3) `extract_Relations`, a classmethod that converts a list of dictionaries of
    relations into `Relation` class objects.

    `schema` details:  `schema` contains a lot of stuff that won't change between
    datasets, so you can copy and paste what I have into your template files. I
    tried to make things easy for you by specifying as additional attributes the components
    of `schema` uses*.  So you should fill in:

    - self.function_name: a valid python name for a function (no spaces)
    - self.instructions: the basic instructions you provide to GPT-4 explaining the task
    - self.properties: a dictionary that contains a key for each entity type and one for relation_type
    (if necessary)*. Each value for a key should be a small dictionary with keys:
        - type: "string"
        - description: a string describing the object.  For an entity type, that will probably be
        very simple. For relation type, you'll want to describe the types of relations allowed. You
        probably don't need to repeat the same descriptions here and in self.instructions.
    - self.required: a list of the components of a relation.  For datasets with multiple relation
    types, a relation type will be needed to be included as well.

    """

    function_name = "extract_relations"
    instructions = "Extracts a list of drug-induced adverse effects relations from a text."

    properties = {
        "drug": {
            "type": "string",
            "description": "drug"
        },
        "ade": {
            "type": "string",
            "description": "adverse effect"
        }
    }

    required = ["drug", "ade"]

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
        relations = set()
        for el in relation_list:
            drug = Entity({el['drug']}, 'drug')
            ade = Entity({el['ade']}, 'ade')
            relation = Relation({drug, ade}, 'ade')
            relations.add(relation)

        return relations

    @classmethod
    def make_prompt(cls, example):
        # system_content = ("You are an advanced relation extractor specialized in pharmacological texts. Your task is "
        #                   "to analyze the sentence and extract all relationships between drugs and their "
        #                   "adverse drug effects (ADEs). Each sentence may contain multiple drug-ADE pairs. "
        #                   "Note that the drug or ADE mentions extracted should have appeared in the original input "
        #                   "text. Carefully identify each drug and its corresponding ADE, and extract them as a list "
        #                   "within a python dictionary, i.e., {'relations': [{'drug': 'drug1', 'ade': 'ade1'}, ...]}")

        system_content = system_prompt

        user_content = f'Abstract: {example.text}'

        system = {'role': 'system',
                  'content': system_content}

        user = {'role': 'user',
                'content': user_content}

        messages = [system, user]

        return messages


class AdeTemplateJSON:

    @classmethod
    def extract_relations(cls, relation_list):
        relations = set()
        relation_list = recursive_lowercase(relation_list)
        for el in relation_list:
            drug = Entity({el['drug']}, 'drug')
            ade = Entity({el['ade']}, 'ade')
            relation = Relation({drug, ade}, 'ade')
            relations.add(relation)

        return relations

    @classmethod
    def make_prompt(cls, example):
        system_content = system_prompt
        user_content = f'Abstract: {example.text}'
        system = {'role': 'system',
                  'content': "You are an advanced relation extractor specialized in pharmacological texts designed to "
                             "output JSON. " + system_content}
        user = {'role': 'user',
                'content': user_content}
        messages = [system, user]

        return messages

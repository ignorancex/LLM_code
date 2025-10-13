#%% libraries
from universal_classes import Entity, Relation  

#%% Templates
class Template1:
        
    ''' 
    A template has three components:
    (1) `schema`, an attribute that contains the general instructions for JSON in GPT-4.
    (2) `make_prompt`, a classmethod that converts an Example into text for GPT-4.
    (3) `extract_Relations`, a classmethod that converts a list of dictionaries of
    relations into `Relation` class objects.
    
    '''
        
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
        
        system_content = '''You are a biomedical researcher interested in diseases induced by drugs/chemicals. When presented with a PubMed abstract, you find all chemical-disease pairs in which the chemical induces the disease, and extract them as a list within a python dictionary, i.e., {'relations': [{'chemical': 'chemical1', 'disease': 'disease1'}, ...]}'''
        
        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]

        return messages

class Template1_json:
        
    ''' 
    A template has three components:
    (1) `schema`, an attribute that contains the general instructions for JSON in GPT-4.
    (2) `make_prompt`, a classmethod that converts an Example into text for GPT-4.
    (3) `extract_Relations`, a classmethod that converts a list of dictionaries of
    relations into `Relation` class objects.
    
    '''
        
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
        
        system_content = '''You are a biomedical researcher interested in diseases induced by drugs/chemicals. When presented with a PubMed abstract, you find all chemical-disease pairs in which the chemical induces the disease, and extract them as a list within a json, i.e., {"relations": [{"chemical": "chemical1", "disease": "disease1"}, ...]}'''
        
        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]

        return messages
#%%    
class Template2:


    ''' 
    Not describing json output.
    '''
        
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
        
        system_content = "You are a biomedical researcher interested in diseases induced by drugs/chemicals. When presented with a PubMed abstract, you find all chemical-disease pairs in which the chemical induces the disease, and extract them as relations."
        
        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]

        return messages
                   

# %%

class Template3:


    ''' 
    Generic system prompt
    '''
        
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
        
        system_content = "You are a helpful assisstant."
        
        user_content = '''When presented with a PubMed abstract, you find all chemical-disease pairs in which the chemical induces the disease, and extract them as relations.''' + f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]

        return messages
    
class Template4:


    ''' 
    No system prompt
    '''
        
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
        
        system_content = '''You are a biomedical researcher interested in diseases induced by drugs/chemicals. When presented with a PubMed abstract, you find all chemical-disease pairs in which the chemical induces the disease, and extract them as relations.'''
        
        user_content = system_content + f'Title: {example.title}\n\n Abstract: {example.text}'
         
        user = {'role': 'user',
                'content': user_content}
        
        messages = [user]

        return messages
    
class Template5:
        
    ''' 
    A template has three components:
    (1) `schema`, an attribute that contains the general instructions for JSON in GPT-4.
    (2) `make_prompt`, a classmethod that converts an Example into text for GPT-4.
    (3) `extract_Relations`, a classmethod that converts a list of dictionaries of
    relations into `Relation` class objects.
    
    '''
        
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
        
        system_content = '''You are a biomedical researcher interested in diseases induced by drugs/chemicals. When presented with a PubMed abstract, you find all chemical-disease pairs in which the chemical induces the disease, and extract them as a list within a python dictionary, i.e., {'relations': [{'chemical': 'chemical1', 'disease': 'disease1'}, ...]}
         
        Here is an example of what your output should could look like:

        {'relations': [{'chemical': 'acetaminophen', 'disease': 'liver disease'}, {'chemical': 'valproic acid', 'disease': 'pancreatitis'}]}'''
        
        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]

        return messages
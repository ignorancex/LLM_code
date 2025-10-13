from universal_classes import Entity, Relation  
from copy import deepcopy               

#%%
class Template1:
    
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
    
    relation_type = {
        "type": "string",
        "description": "A biomedical relationship that holds between two entities, either Positive_Correlation, Negative_Correlation, Association, Bind, Cotreatment, Drug_Interaction, Comparison, or Conversion"
    }
    
    relation = {
        "type": "object",
        "description": "Two biomedical entities and a relationship that holds between them.",
        "properties": {
            "entity1": entity1,
            "entity2": entity2,
            "relation_type": relation_type
            },
        "required": ["entity1", "entity2", "relation_type"]
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
            relation = Relation({entity1, entity2}, el['relation_type'])
            relations.add(relation)

        return relations

    @classmethod
    def make_prompt(cls, example):
        
        system_content = '''You are a biomedical researcher interested in relationships holding between genes, gene variants, chemicals/drugs, and diseases. When presented with a PubMed abstract, you find all pairs of entities belonging to a set of predefined relation types, and extract them as a list within a python dictionary, i.e., {'relations': [{'entity1': {'text': 'text1', 'entity_type': 'entity_type1'}, 'entity2': {'text': 'text2', 'entity_type': 'entity_type2'}, 'relation_type': 'relation_type1'}, ...]}
        
The entity types must either be GeneOrGeneProduct, SequenceVariant, ChemicalEntity, or DiseaseOrPhenotypicFeature, and the relation types must be either Positive_Correlation, Negative_Correlation, Association, Bind, Cotreatment, Drug_Interaction, Comparison, or Conversion.  Relation types may only be valid for certain combinations of entity types and may have somewhat different meanings based on the pair of entity types.  Below are guidelines for extracting relations.
        
Disease-Chemical Relations:

    Positive_Correlation: Chemicals may induce diseases, increase disease risk, or their levels may correlate with disease risk.

    Negative_Correlation: Chemicals used as drugs may treat diseases or decrease disease susceptibility.

    Association: Captures relationships not clearly defined as positive or negative correlations, such as drugs with potential safety concerns.

Disease-Gene Relations:

    Positive_Correlation: Overexpression or side effects of proteins (from genes) may cause diseases.

    Negative_Correlation: Proteins used as drugs may treat diseases or their absence may cause diseases.

    Association: Includes functional genes preventing diseases and other associations not falling under positive/negative correlations.

Disease-Variant Relations:

    Positive_Correlation: Variants may increase disease risk, contribute to disease susceptibility, or cause protein deficiencies leading to diseases.

    Negative_Correlation: Variants may decrease disease risk.

    Association: Includes variants associated with disease prevalence and those that cannot be categorized as causing the disease.

Gene-Gene Relations:

    Positive_Correlation: Genes may show positive correlations in expression or regulatory functions.

    Negative_Correlation: Genes may show negative correlations in expression or regulatory functions.
    
    Association: Modifications like phosphorylation or other associations that cannot be categorized differently.
    
    Bind: Physical interactions between proteins, including protein binding at gene promoters.

Gene-Chemical Relations:

    Positive_Correlation: Chemicals may cause higher gene expression or gene variants may trigger chemical adverse effects.

    Negative_Correlation: Chemicals may cause lower gene expression or gene variants may confer resistance to chemicals.

    Association: Non-specific associations and binding interactions between chemicals and gene promoters.

Chemical-Chemical Relations:

    Positive_Correlation: One chemical may increase the sensitivity or effectiveness of another.

    Negative_Correlation: One chemical may decrease the sensitivity or side effects of another.

    Association: Includes chemical conversions and non-specific associations.
    
    Drug_Interaction: Pharmacodynamic interactions between chemicals or drugs.

    Cotreatment: Combination therapy using multiple chemicals.
    
    Conversion: One chemical converting to another.

Chemical-Variant Relations:

    Positive_Correlation: Chemicals may cause higher expression of a gene variant or increase sensitivity due to a variant.

    Negative_Correlation: Chemicals may decrease gene expression due to a variant or the variant may confer resistance.

    Association: Captures relationships not defined as positive/negative correlations, like variants on chemical binding sites.'''
        
        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]
        
        return messages

#%%
class Template2:
    
    '''Making connection between formal entity types and informal entity types clearer.'''

    text = {
        "type": "string",
        "description": "Text of entity"
        }
    
    entity_type = {
        "type": "string",
        "description": "One of the following categories: gene, gene variant, chemical, or disease"
        }
    
    entity1 = {"type": "object",
               "description": "A biomedical entity that is either a gene, gene variant, chemical, or disease",
               "properties": {
                   "text": text,
                   "entity_type": entity_type
                   },
               "required": ["text", "entity_type"]
        }
    
    entity2 = deepcopy(entity1)
    
    relation_type = {
        "type": "string",
        "description": "A biomedical relationship that holds between two entities, either positive correlation, negative correlation, association, bind, cause, co-treatment, drug-interaction, comparison"
    }
    
    relation = {
        "type": "object",
        "description": "Two biomedical entities and a relationship that holds between them.",
        "properties": {
            "entity1": entity1,
            "entity2": entity2,
            "relation_type": relation_type
            },
        "required": ["entity1", "entity2", "relation_type"]
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
            relation = Relation({entity1, entity2}, el['relation_type'])
            relations.add(relation)

        return relations

    @classmethod
    def make_prompt(cls, example):
        
        system_content = '''You are a biomedical researcher interested in relationships holding between genes, gene variants, chemicals/drugs, and diseases. When presented with a PubMed abstract, you find all pairs of entities belonging to a set of predefined relation types, and extract them as a list within a python dictionary, i.e., {'relations': [{'entity1': {'text': 'text1', 'entity_type': 'entity_type1'}, 'entity2': {'text': 'text2', 'entity_type': 'entity_type2'}, 'relation_type': 'relation_type1'}, ...]}
        
The entity types must either be GeneOrGeneProduct (for a gene), SequenceVariant (for a gene variant), ChemicalEntity (for a chemical/drug), or DiseaseOrPhenotypicFeature (for a disease), and the relation types must be either Positive_Correlation, Negative_Correlation, Association, Bind, Cotreatment, Drug_Interaction, Comparison, or Conversion.  Relation types may only be valid for certain combinations of entity types and may have somewhat different meanings based on the pair of entity types.  Below are guidelines for extracting relations.
        
Disease-Chemical Relations:

    Positive_Correlation: Chemicals may induce diseases, increase disease risk, or their levels may correlate with disease risk.

    Negative_Correlation: Chemicals used as drugs may treat diseases or decrease disease susceptibility.

    Association: Captures relationships not clearly defined as positive or negative correlations, such as drugs with potential safety concerns.

Disease-Gene Relations:

    Positive_Correlation: Overexpression or side effects of proteins (from genes) may cause diseases.

    Negative_Correlation: Proteins used as drugs may treat diseases or their absence may cause diseases.

    Association: Includes functional genes preventing diseases and other associations not falling under positive/negative correlations.

Disease-Variant Relations:

    Positive_Correlation: Variants may increase disease risk, contribute to disease susceptibility, or cause protein deficiencies leading to diseases.

    Negative_Correlation: Variants may decrease disease risk.

    Association: Includes variants associated with disease prevalence and those that cannot be categorized as causing the disease.

Gene-Gene Relations:

    Positive_Correlation: Genes may show positive correlations in expression or regulatory functions.

    Negative_Correlation: Genes may show negative correlations in expression or regulatory functions.
    
    Association: Modifications like phosphorylation or other associations that cannot be categorized differently.
    
    Bind: Physical interactions between proteins, including protein binding at gene promoters.

Gene-Chemical Relations:

    Positive_Correlation: Chemicals may cause higher gene expression or gene variants may trigger chemical adverse effects.

    Negative_Correlation: Chemicals may cause lower gene expression or gene variants may confer resistance to chemicals.

    Association: Non-specific associations and binding interactions between chemicals and gene promoters.

Chemical-Chemical Relations:

    Positive_Correlation: One chemical may increase the sensitivity or effectiveness of another.

    Negative_Correlation: One chemical may decrease the sensitivity or side effects of another.

    Association: Includes chemical conversions and non-specific associations.
    
    Drug_Interaction: Pharmacodynamic interactions between chemicals or drugs.

    Cotreatment: Combination therapy using multiple chemicals.
    
    Conversion: One chemical converting to another.

Chemical-Variant Relations:

    Positive_Correlation: Chemicals may cause higher expression of a gene variant or increase sensitivity due to a variant.

    Negative_Correlation: Chemicals may decrease gene expression due to a variant or the variant may confer resistance.

    Association: Captures relationships not defined as positive/negative correlations, like variants on chemical binding sites.'''
        
        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]
        
        return messages         

#%%
class Template2b:
    
    '''Making connection between formal entity types and informal entity types clearer.'''

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
    
    relation_type = {
        "type": "string",
        "description": "A biomedical relationship that holds between two entities, either Positive_Correlation, Negative_Correlation, Association, Bind, Cotreatment, Drug_Interaction, Comparison, or Conversion"
    }
    
    relation = {
        "type": "object",
        "description": "Two biomedical entities and a relationship that holds between them.",
        "properties": {
            "entity1": entity1,
            "entity2": entity2,
            "relation_type": relation_type
            },
        "required": ["entity1", "entity2", "relation_type"]
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
            relation = Relation({entity1, entity2}, el['relation_type'])
            relations.add(relation)

        return relations

    @classmethod
    def make_prompt(cls, example):
        
        system_content = '''You are a biomedical researcher interested in relationships holding between genes, gene variants, chemicals/drugs, and diseases. When presented with a PubMed abstract, you find all pairs of entities belonging to a set of predefined relation types, and extract them as a list within a python dictionary, i.e., {'relations': [{'entity1': {'text': 'text1', 'entity_type': 'entity_type1'}, 'entity2': {'text': 'text2', 'entity_type': 'entity_type2'}, 'relation_type': 'relation_type1'}, ...]}
        
The entity types must either be GeneOrGeneProduct (for a gene), SequenceVariant (for a gene variant), ChemicalEntity (for a chemical/drug), or DiseaseOrPhenotypicFeature (for a disease), and the relation types must be either Positive_Correlation, Negative_Correlation, Association, Bind, Cotreatment, Drug_Interaction, Comparison, or Conversion.  Relation types may only be valid for certain combinations of entity types and may have somewhat different meanings based on the pair of entity types.  Below are guidelines for extracting relations.
        
Disease-Chemical Relations:

    Positive_Correlation: Chemicals may induce diseases, increase disease risk, or their levels may correlate with disease risk.

    Negative_Correlation: Chemicals used as drugs may treat diseases or decrease disease susceptibility.

    Association: Captures relationships not clearly defined as positive or negative correlations, such as drugs with potential safety concerns.

Disease-Gene Relations:

    Positive_Correlation: Overexpression or side effects of proteins (from genes) may cause diseases.

    Negative_Correlation: Proteins used as drugs may treat diseases or their absence may cause diseases.

    Association: Includes functional genes preventing diseases and other associations not falling under positive/negative correlations.

Disease-Variant Relations:

    Positive_Correlation: Variants may increase disease risk, contribute to disease susceptibility, or cause protein deficiencies leading to diseases.

    Negative_Correlation: Variants may decrease disease risk.

    Association: Includes variants associated with disease prevalence and those that cannot be categorized as causing the disease.

Gene-Gene Relations:

    Positive_Correlation: Genes may show positive correlations in expression or regulatory functions.

    Negative_Correlation: Genes may show negative correlations in expression or regulatory functions.
    
    Association: Modifications like phosphorylation or other associations that cannot be categorized differently.
    
    Bind: Physical interactions between proteins, including protein binding at gene promoters.

Gene-Chemical Relations:

    Positive_Correlation: Chemicals may cause higher gene expression or gene variants may trigger chemical adverse effects.

    Negative_Correlation: Chemicals may cause lower gene expression or gene variants may confer resistance to chemicals.

    Association: Non-specific associations and binding interactions between chemicals and gene promoters.

Chemical-Chemical Relations:

    Positive_Correlation: One chemical may increase the sensitivity or effectiveness of another.

    Negative_Correlation: One chemical may decrease the sensitivity or side effects of another.

    Association: Includes chemical conversions and non-specific associations.
    
    Drug_Interaction: Pharmacodynamic interactions between chemicals or drugs.

    Cotreatment: Combination therapy using multiple chemicals.
    
    Conversion: One chemical converting to another.

Chemical-Variant Relations:

    Positive_Correlation: Chemicals may cause higher expression of a gene variant or increase sensitivity due to a variant.

    Negative_Correlation: Chemicals may decrease gene expression due to a variant or the variant may confer resistance.

    Association: Captures relationships not defined as positive/negative correlations, like variants on chemical binding sites.'''
        
        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]
        
        return messages  

#%%
class Template3:
    
    '''Removing drug interaction and conversion relation types, since they are extremely rare.'''

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
    
    relation_type = {
        "type": "string",
        "description": "A biomedical relationship that holds between two entities, either Positive_Correlation, Negative_Correlation, Association, Bind, Cotreatment, or Comparison"
    }
    
    relation = {
        "type": "object",
        "description": "Two biomedical entities and a relationship that holds between them.",
        "properties": {
            "entity1": entity1,
            "entity2": entity2,
            "relation_type": relation_type
            },
        "required": ["entity1", "entity2", "relation_type"]
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
            relation = Relation({entity1, entity2}, el['relation_type'])
            relations.add(relation)

        return relations

    @classmethod
    def make_prompt(cls, example):
        
        system_content = '''You are a biomedical researcher interested in relationships holding between genes, gene variants, chemicals/drugs, and diseases. When presented with a PubMed abstract, you find all pairs of entities belonging to a set of predefined relation types, and extract them as a list within a python dictionary, i.e., {'relations': [{'entity1': {'text': 'text1', 'entity_type': 'entity_type1'}, 'entity2': {'text': 'text2', 'entity_type': 'entity_type2'}, 'relation_type': 'relation_type1'}, ...]}
        
The entity types must either be GeneOrGeneProduct (for a gene), SequenceVariant (for a gene variant), ChemicalEntity (for a chemical/drug), or DiseaseOrPhenotypicFeature (for a disease), and the relation types must be either Positive_Correlation, Negative_Correlation, Association, Bind, Cotreatment, or Comparison.  Relation types may only be valid for certain combinations of entity types and may have somewhat different meanings based on the pair of entity types.  Below are guidelines for extracting relations.
        
Disease-Chemical Relations:

    Positive_Correlation: Chemicals may induce diseases, increase disease risk, or their levels may correlate with disease risk.

    Negative_Correlation: Chemicals used as drugs may treat diseases or decrease disease susceptibility.

    Association: Captures relationships not clearly defined as positive or negative correlations, such as drugs with potential safety concerns.

Disease-Gene Relations:

    Positive_Correlation: Overexpression or side effects of proteins (from genes) may cause diseases.

    Negative_Correlation: Proteins used as drugs may treat diseases or their absence may cause diseases.

    Association: Includes functional genes preventing diseases and other associations not falling under positive/negative correlations.

Disease-Variant Relations:

    Positive_Correlation: Variants may increase disease risk, contribute to disease susceptibility, or cause protein deficiencies leading to diseases.

    Negative_Correlation: Variants may decrease disease risk.

    Association: Includes variants associated with disease prevalence and those that cannot be categorized as causing the disease.

Gene-Gene Relations:

    Positive_Correlation: Genes may show positive correlations in expression or regulatory functions.

    Negative_Correlation: Genes may show negative correlations in expression or regulatory functions.
    
    Association: Modifications like phosphorylation or other associations that cannot be categorized differently.
    
    Bind: Physical interactions between proteins, including protein binding at gene promoters.

Gene-Chemical Relations:

    Positive_Correlation: Chemicals may cause higher gene expression or gene variants may trigger chemical adverse effects.

    Negative_Correlation: Chemicals may cause lower gene expression or gene variants may confer resistance to chemicals.

    Association: Non-specific associations and binding interactions between chemicals and gene promoters.

Chemical-Chemical Relations:

    Positive_Correlation: One chemical may increase the sensitivity or effectiveness of another.

    Negative_Correlation: One chemical may decrease the sensitivity or side effects of another.

    Association: Includes chemical conversions and non-specific associations.
    
    Cotreatment: Combination therapy using multiple chemicals.
    
Chemical-Variant Relations:

    Positive_Correlation: Chemicals may cause higher expression of a gene variant or increase sensitivity due to a variant.

    Negative_Correlation: Chemicals may decrease gene expression due to a variant or the variant may confer resistance.

    Association: Captures relationships not defined as positive/negative correlations, like variants on chemical binding sites.'''
        
        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]
        
        return messages      
    
#%%
class Template4:
    
    '''Keeping only positive correlation, negative correlation, and association relation types'''

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
    
    relation_type = {
        "type": "string",
        "description": "A biomedical relationship that holds between two entities, either Positive_Correlation, Negative_Correlation, or Association"
    }
    
    relation = {
        "type": "object",
        "description": "Two biomedical entities and a relationship that holds between them.",
        "properties": {
            "entity1": entity1,
            "entity2": entity2,
            "relation_type": relation_type
            },
        "required": ["entity1", "entity2", "relation_type"]
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
            relation = Relation({entity1, entity2}, el['relation_type'])
            relations.add(relation)

        return relations

    @classmethod
    def make_prompt(cls, example):
        
        system_content = '''You are a biomedical researcher interested in relationships holding between genes, gene variants, chemicals/drugs, and diseases. When presented with a PubMed abstract, you find all pairs of entities belonging to a set of predefined relation types, and extract them as a list within a python dictionary, i.e., {'relations': [{'entity1': {'text': 'text1', 'entity_type': 'entity_type1'}, 'entity2': {'text': 'text2', 'entity_type': 'entity_type2'}, 'relation_type': 'relation_type1'}, ...]}
        
The entity types must either be GeneOrGeneProduct (for a gene), SequenceVariant (for a gene variant), ChemicalEntity (for a chemical/drug), or DiseaseOrPhenotypicFeature (for a disease), and the relation types must be either Positive_Correlation, Negative_Correlation, or Association.  Relation types may only be valid for certain combinations of entity types and may have somewhat different meanings based on the pair of entity types.  Below are guidelines for extracting relations.
        
Disease-Chemical Relations:

    Positive_Correlation: Chemicals may induce diseases, increase disease risk, or their levels may correlate with disease risk.

    Negative_Correlation: Chemicals used as drugs may treat diseases or decrease disease susceptibility.

    Association: Captures relationships not clearly defined as positive or negative correlations, such as drugs with potential safety concerns.

Disease-Gene Relations:

    Positive_Correlation: Overexpression or side effects of proteins (from genes) may cause diseases.

    Negative_Correlation: Proteins used as drugs may treat diseases or their absence may cause diseases.

    Association: Includes functional genes preventing diseases and other associations not falling under positive/negative correlations.

Disease-Variant Relations:

    Positive_Correlation: Variants may increase disease risk, contribute to disease susceptibility, or cause protein deficiencies leading to diseases.

    Negative_Correlation: Variants may decrease disease risk.

    Association: Includes variants associated with disease prevalence and those that cannot be categorized as causing the disease.

Gene-Gene Relations:

    Positive_Correlation: Genes may show positive correlations in expression or regulatory functions.

    Negative_Correlation: Genes may show negative correlations in expression or regulatory functions.
    
    Association: Modifications like phosphorylation or other associations that cannot be categorized differently.
    
Gene-Chemical Relations:

    Positive_Correlation: Chemicals may cause higher gene expression or gene variants may trigger chemical adverse effects.

    Negative_Correlation: Chemicals may cause lower gene expression or gene variants may confer resistance to chemicals.

    Association: Non-specific associations and binding interactions between chemicals and gene promoters.

Chemical-Chemical Relations:

    Positive_Correlation: One chemical may increase the sensitivity or effectiveness of another.

    Negative_Correlation: One chemical may decrease the sensitivity or side effects of another.

    Association: Includes chemical conversions and non-specific associations.
        
Chemical-Variant Relations:

    Positive_Correlation: Chemicals may cause higher expression of a gene variant or increase sensitivity due to a variant.

    Negative_Correlation: Chemicals may decrease gene expression due to a variant or the variant may confer resistance.

    Association: Captures relationships not defined as positive/negative correlations, like variants on chemical binding sites.'''
        
        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]
        
        return messages   
    
#%%
class Template5:
    
    '''Trying to get GPT to increase # of predicted relations.'''

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
    
    relation_type = {
        "type": "string",
        "description": "A biomedical relationship that holds between two entities, either Positive_Correlation, Negative_Correlation, Association, Bind, Cotreatment, Drug_Interaction, Comparison, or Conversion"
    }
    
    relation = {
        "type": "object",
        "description": "Two biomedical entities and a relationship that holds between them.",
        "properties": {
            "entity1": entity1,
            "entity2": entity2,
            "relation_type": relation_type
            },
        "required": ["entity1", "entity2", "relation_type"]
        }
    
    relations = {
        "type": "array",
        "description": "List of biomedical relations, typically of length 5-15, sometimes up to 50, and rarely 0",
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
            relation = Relation({entity1, entity2}, el['relation_type'])
            relations.add(relation)

        return relations

    @classmethod
    def make_prompt(cls, example):
        
        system_content = '''You are a biomedical researcher interested in relationships holding between genes, gene variants, chemicals/drugs, and diseases. When presented with a PubMed abstract, you find all pairs of entities belonging to a set of predefined relation types, and extract them as a list within a python dictionary, i.e., {'relations': [{'entity1': {'text': 'text1', 'entity_type': 'entity_type1'}, 'entity2': {'text': 'text2', 'entity_type': 'entity_type2'}, 'relation_type': 'relation_type1'}, ...]}. A typical abstract contains 5-15 relations, but sometimes up to 50; 0 relations is exceedingly rare.
        
The entity types must either be GeneOrGeneProduct (for a gene), SequenceVariant (for a gene variant), ChemicalEntity (for a chemical/drug), or DiseaseOrPhenotypicFeature (for a disease), and the relation types must be either Positive_Correlation, Negative_Correlation, Association, Bind, Cotreatment, Drug_Interaction, Comparison, or Conversion.  Relation types may only be valid for certain combinations of entity types and may have somewhat different meanings based on the pair of entity types.  Below are guidelines for extracting relations.
        
Disease-Chemical Relations:

    Positive_Correlation: Chemicals may induce diseases, increase disease risk, or their levels may correlate with disease risk.

    Negative_Correlation: Chemicals used as drugs may treat diseases or decrease disease susceptibility.

    Association: Captures relationships not clearly defined as positive or negative correlations, such as drugs with potential safety concerns.

Disease-Gene Relations:

    Positive_Correlation: Overexpression or side effects of proteins (from genes) may cause diseases.

    Negative_Correlation: Proteins used as drugs may treat diseases or their absence may cause diseases.

    Association: Includes functional genes preventing diseases and other associations not falling under positive/negative correlations.

Disease-Variant Relations:

    Positive_Correlation: Variants may increase disease risk, contribute to disease susceptibility, or cause protein deficiencies leading to diseases.

    Negative_Correlation: Variants may decrease disease risk.

    Association: Includes variants associated with disease prevalence and those that cannot be categorized as causing the disease.

Gene-Gene Relations:

    Positive_Correlation: Genes may show positive correlations in expression or regulatory functions.

    Negative_Correlation: Genes may show negative correlations in expression or regulatory functions.
    
    Association: Modifications like phosphorylation or other associations that cannot be categorized differently.
    
    Bind: Physical interactions between proteins, including protein binding at gene promoters.

Gene-Chemical Relations:

    Positive_Correlation: Chemicals may cause higher gene expression or gene variants may trigger chemical adverse effects.

    Negative_Correlation: Chemicals may cause lower gene expression or gene variants may confer resistance to chemicals.

    Association: Non-specific associations and binding interactions between chemicals and gene promoters.

Chemical-Chemical Relations:

    Positive_Correlation: One chemical may increase the sensitivity or effectiveness of another.

    Negative_Correlation: One chemical may decrease the sensitivity or side effects of another.

    Association: Includes chemical conversions and non-specific associations.
    
    Drug_Interaction: Pharmacodynamic interactions between chemicals or drugs.

    Cotreatment: Combination therapy using multiple chemicals.
    
    Conversion: One chemical converting to another.

Chemical-Variant Relations:

    Positive_Correlation: Chemicals may cause higher expression of a gene variant or increase sensitivity due to a variant.

    Negative_Correlation: Chemicals may decrease gene expression due to a variant or the variant may confer resistance.

    Association: Captures relationships not defined as positive/negative correlations, like variants on chemical binding sites.

    '''
        
        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]
        
        return messages  
    
class MultiTemplate:
    
    '''Separating relations by pairs of entity types.'''

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
    
    relation_type = {
        "type": "string",
        "description": "A biomedical relationship that holds between two entities, either Positive_Correlation, Negative_Correlation, Association, Bind, Cotreatment, Drug_Interaction, Comparison, or Conversion"
    }
    
    relation = {
        "type": "object",
        "description": "Two biomedical entities and a relationship that holds between them.",
        "properties": {
            "entity1": entity1,
            "entity2": entity2,
            "relation_type": relation_type
            },
        "required": ["entity1", "entity2", "relation_type"]
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
            relation = Relation({entity1, entity2}, el['relation_type'])
            relations.add(relation)

        return relations
    
    @classmethod
    def make_prompt(cls, example):
        
        system_content = '''You are a biomedical researcher interested in relationships holding between genes, gene variants, chemicals/drugs, and diseases. When presented with a PubMed abstract, you find all pairs of entities belonging to a set of predefined relation types, and extract them as a list within a python dictionary, i.e., {'relations': [{'entity1': {'text': 'text1', 'entity_type': 'entity_type1'}, 'entity2': {'text': 'text2', 'entity_type': 'entity_type2'}, 'relation_type': 'relation_type1'}, ...]}
        
        The entity types must either be GeneOrGeneProduct (for a gene), SequenceVariant (for a gene variant), ChemicalEntity (for a chemical/drug), or DiseaseOrPhenotypicFeature (for a disease), and the relation types must be either Positive_Correlation, Negative_Correlation, Association, Bind, Cotreatment, Drug_Interaction, Comparison, or Conversion.  Relation types may only be valid for certain combinations of entity types and may have somewhat different meanings based on the pair of entity types.  Below are guidelines for extracting relations.
        
        Disease-Chemical Relations:

            Positive_Correlation: Chemicals may induce diseases, increase disease risk, or their levels may correlate with disease risk.

            Negative_Correlation: Chemicals used as drugs may treat diseases or decrease disease susceptibility.

            Association: Captures relationships not clearly defined as positive or negative correlations, such as drugs with potential safety concerns.

        Disease-Gene Relations:

            Positive_Correlation: Overexpression or side effects of proteins (from genes) may cause diseases.

            Negative_Correlation: Proteins used as drugs may treat diseases or their absence may cause diseases.

            Association: Includes functional genes preventing diseases and other associations not falling under positive/negative correlations.

        Disease-Variant Relations:

            Positive_Correlation: Variants may increase disease risk, contribute to disease susceptibility, or cause protein deficiencies leading to diseases.

            Negative_Correlation: Variants may decrease disease risk.

            Association: Includes variants associated with disease prevalence and those that cannot be categorized as causing the disease.

        Gene-Gene Relations:

            Positive_Correlation: Genes may show positive correlations in expression or regulatory functions.

            Negative_Correlation: Genes may show negative correlations in expression or regulatory functions.
            
            Association: Modifications like phosphorylation or other associations that cannot be categorized differently.
            
            Bind: Physical interactions between proteins, including protein binding at gene promoters.

        Gene-Chemical Relations:

            Positive_Correlation: Chemicals may cause higher gene expression or gene variants may trigger chemical adverse effects.

            Negative_Correlation: Chemicals may cause lower gene expression or gene variants may confer resistance to chemicals.

            Association: Non-specific associations and binding interactions between chemicals and gene promoters.

        Chemical-Chemical Relations:

            Positive_Correlation: One chemical may increase the sensitivity or effectiveness of another.

            Negative_Correlation: One chemical may decrease the sensitivity or side effects of another.

            Association: Includes chemical conversions and non-specific associations.
            
            Drug_Interaction: Pharmacodynamic interactions between chemicals or drugs.

            Cotreatment: Combination therapy using multiple chemicals.
            
            Conversion: One chemical converting to another.

        Chemical-Variant Relations:

            Positive_Correlation: Chemicals may cause higher expression of a gene variant or increase sensitivity due to a variant.

            Negative_Correlation: Chemicals may decrease gene expression due to a variant or the variant may confer resistance.

            Association: Captures relationships not defined as positive/negative correlations, like variants on chemical binding sites.'''
                
        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]
        
        return messages  

class MultiTemplateA:
    
    '''Separating relations by pairs of entity types. Disease-Chemical Relations'''

    text = {
        "type": "string",
        "description": "Text of entity"
        }
    
    entity_type = {
        "type": "string",
        "description": "ChemicalEntity or DiseaseOrPhenotypicFeature"
        }
    
    entity1 = {"type": "object",
               "description": "A biomedical entity that is either a ChemicalEntity or DiseaseOrPhenotypicFeature",
               "properties": {
                   "text": text,
                   "entity_type": entity_type
                   },
               "required": ["text", "entity_type"]
        }
    
    entity2 = deepcopy(entity1)
    
    relation_type = {
        "type": "string",
        "description": "A biomedical relationship that holds between two entities, either Positive_Correlation, Negative_Correlation, or Association."
    }
    
    relation = {
        "type": "object",
        "description": "Two biomedical entities and a relationship that holds between them.",
        "properties": {
            "entity1": entity1,
            "entity2": entity2,
            "relation_type": relation_type
            },
        "required": ["entity1", "entity2", "relation_type"]
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
            relation = Relation({entity1, entity2}, el['relation_type'])
            relations.add(relation)

        return relations
    
    @classmethod
    def make_prompt(cls, example):
        
        system_content = '''You are a biomedical researcher interested in relationships holding between diseases and chemicals. When presented with a PubMed abstract, you find all disease-chemical pairs of entities exhibiting one of a set of predefined relation types, and extract them as a list within a python dictionary, i.e., {'relations': [{'entity1': {'text': 'text1', 'entity_type': 'entity_type1'}, 'entity2': {'text': 'text2', 'entity_type': 'entity_type2'}, 'relation_type': 'relation_type1'}, ...]}
        
        The entity types must either be DiseaseOrPhenotypicFeature (for a disease) or ChemicalEntity (for a chemical/drug), and the relation types must be either Positive_Correlation, Negative_Correlation, or Association. Guidelines for extracting relation types are given below.
        
        Positive_Correlation: Chemicals may induce diseases, increase disease risk, or their levels may correlate with disease risk.

        Negative_Correlation: Chemicals used as drugs may treat diseases or decrease disease susceptibility.

        Association: Captures relationships not clearly defined as positive or negative correlations, such as drugs with potential safety concerns.'''
        
        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]
        
        return messages  

class MultiTemplateB:

    '''Separating relations by pairs of entity types. Disease-Gene'''

    text = {
        "type": "string",
        "description": "Text of entity"
        }
    
    entity_type = {
        "type": "string",
        "description": "GeneOrGeneProduct or DiseaseOrPhenotypicFeature"
        }
    
    entity1 = {"type": "object",
               "description": "A biomedical entity that is either a GeneOrGeneProduct or DiseaseOrPhenotypicFeature",
               "properties": {
                   "text": text,
                   "entity_type": entity_type
                   },
               "required": ["text", "entity_type"]
        }
    
    entity2 = deepcopy(entity1)
    
    relation_type = {
        "type": "string",
        "description": "A biomedical relationship that holds between two entities, either Positive_Correlation, Negative_Correlation, or Association."
    }
    
    relation = {
        "type": "object",
        "description": "Two biomedical entities and a relationship that holds between them.",
        "properties": {
            "entity1": entity1,
            "entity2": entity2,
            "relation_type": relation_type
            },
        "required": ["entity1", "entity2", "relation_type"]
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
            relation = Relation({entity1, entity2}, el['relation_type'])
            relations.add(relation)

        return relations
    

    @classmethod
    def make_prompt(cls, example):
        
        system_content = '''You are a biomedical researcher interested in relationships holding between diseases and genes. When presented with a PubMed abstract, you find all disease-chemical pairs of entities exhibiting one of a set of predefined relation types, and extract them as a list within a python dictionary, i.e., {'relations': [{'entity1': {'text': 'text1', 'entity_type': 'entity_type1'}, 'entity2': {'text': 'text2', 'entity_type': 'entity_type2'}, 'relation_type': 'relation_type1'}, ...]}
        
        The entity types must either be DiseaseOrPhenotypicFeature (for a disease) or GeneOrGeneProduct (for a gene), and the relation types must be either Positive_Correlation, Negative_Correlation, or Association. Guidelines for extracting relation types are given below.
        
        Positive_Correlation: Overexpression or side effects of proteins (from genes) may cause diseases.

        Negative_Correlation: Proteins used as drugs may treat diseases or their absence may cause diseases.

        Association: Includes functional genes preventing diseases and other associations not falling under positive/negative correlations.'''
        
        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]
        
        return messages  
    
class MultiTemplateC:

    '''Separating relations by pairs of entity types. Disease-Variant Relations'''

    text = {
        "type": "string",
        "description": "Text of entity"
        }
    
    entity_type = {
        "type": "string",
        "description": "GeneOrGeneProduct or SequenceVariant"
        }
    
    entity1 = {"type": "object",
               "description": "A biomedical entity that is either a GeneOrGeneProduct or SequenceVariant",
               "properties": {
                   "text": text,
                   "entity_type": entity_type
                   },
               "required": ["text", "entity_type"]
        }
    
    entity2 = deepcopy(entity1)
    
    relation_type = {
        "type": "string",
        "description": "A biomedical relationship that holds between two entities, either Positive_Correlation, Negative_Correlation, or Association."
    }
    
    relation = {
        "type": "object",
        "description": "Two biomedical entities and a relationship that holds between them.",
        "properties": {
            "entity1": entity1,
            "entity2": entity2,
            "relation_type": relation_type
            },
        "required": ["entity1", "entity2", "relation_type"]
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
            relation = Relation({entity1, entity2}, el['relation_type'])
            relations.add(relation)

        return relations
    

    @classmethod
    def make_prompt(cls, example):
        
        system_content = '''You are a biomedical researcher interested in relationships holding between diseases and gene variants. When presented with a PubMed abstract, you find all disease-variant pairs of entities exhibiting one of a set of predefined relation types, and extract them as a list within a python dictionary, i.e., {'relations': [{'entity1': {'text': 'text1', 'entity_type': 'entity_type1'}, 'entity2': {'text': 'text2', 'entity_type': 'entity_type2'}, 'relation_type': 'relation_type1'}, ...]}
        
        The entity types must either be DiseaseOrPhenotypicFeature (for a disease) or SequenceVariant (for a gene variant), and the relation types must be either Positive_Correlation, Negative_Correlation, or Association. Guidelines for extracting relation types are given below.
        
        Positive_Correlation: Variants may increase disease risk, contribute to disease susceptibility, or cause protein deficiencies leading to diseases.

        Negative_Correlation: Variants may decrease disease risk.

        Association: Includes variants associated with disease prevalence and those that cannot be categorized as causing the disease.'''

        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]
        
        return messages  
    
class MultiTemplateD:

    '''Separating relations by pairs of entity types. Gene-Gene Relations'''

    text = {
        "type": "string",
        "description": "Text of entity"
        }
    
    entity_type = {
        "type": "string",
        "description": "GeneOrGeneProduct"
        }
    
    entity1 = {"type": "object",
               "description": "A biomedical entity that is a GeneOrGeneProduct",
               "properties": {
                   "text": text,
                   "entity_type": entity_type
                   },
               "required": ["text", "entity_type"]
        }
    
    entity2 = deepcopy(entity1)
    
    relation_type = {
        "type": "string",
        "description": "A biomedical relationship that holds between two entities, either Positive_Correlation, Negative_Correlation, Association, or Bind."
    }
    
    relation = {
        "type": "object",
        "description": "Two biomedical entities and a relationship that holds between them.",
        "properties": {
            "entity1": entity1,
            "entity2": entity2,
            "relation_type": relation_type
            },
        "required": ["entity1", "entity2", "relation_type"]
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
            relation = Relation({entity1, entity2}, el['relation_type'])
            relations.add(relation)

        return relations
    

    @classmethod
    def make_prompt(cls, example):
        
        system_content = '''You are a biomedical researcher interested in relationships holding between pairs of genes. When presented with a PubMed abstract, you find all gene-gene pairs of entities exhibiting one of a set of predefined relation types, and extract them as a list within a python dictionary, i.e., {'relations': [{'entity1': {'text': 'text1', 'entity_type': 'entity_type1'}, 'entity2': {'text': 'text2', 'entity_type': 'entity_type2'}, 'relation_type': 'relation_type1'}, ...]}
        
        The entity types must either be GeneOrGeneProduct (for a gene), and the relation types must be either Positive_Correlation, Negative_Correlation, or Association. Guidelines for extracting relation types are given below.
        
        Positive_Correlation: Genes may show positive correlations in expression or regulatory functions.

        Negative_Correlation: Genes may show negative correlations in expression or regulatory functions.
            
        Association: Modifications like phosphorylation or other associations that cannot be categorized differently.
            
        Bind: Physical interactions between proteins, including protein binding at gene promoters.'''

        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]
        
        return messages  
    
class MultiTemplateE:

    '''Separating relations by pairs of entity types. Gene-Chemical Relations'''

    text = {
        "type": "string",
        "description": "Text of entity"
        }
    
    entity_type = {
        "type": "string",
        "description": "GeneOrGeneProduct or ChemicalEntity"
        }
    
    entity1 = {"type": "object",
               "description": "A biomedical entity that is a GeneOrGeneProduct or ChemicalEntity",
               "properties": {
                   "text": text,
                   "entity_type": entity_type
                   },
               "required": ["text", "entity_type"]
        }
    
    entity2 = deepcopy(entity1)
    
    relation_type = {
        "type": "string",
        "description": "A biomedical relationship that holds between two entities, either Positive_Correlation, Negative_Correlation, or Association."
    }
    
    relation = {
        "type": "object",
        "description": "Two biomedical entities and a relationship that holds between them.",
        "properties": {
            "entity1": entity1,
            "entity2": entity2,
            "relation_type": relation_type
            },
        "required": ["entity1", "entity2", "relation_type"]
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
            relation = Relation({entity1, entity2}, el['relation_type'])
            relations.add(relation)

        return relations
    

    @classmethod
    def make_prompt(cls, example):
        
        system_content = '''You are a biomedical researcher interested in relationships holding between genes and chemicals/drugs. When presented with a PubMed abstract, you find all gene-chemical pairs of entities exhibiting one of a set of predefined relation types, and extract them as a list within a python dictionary, i.e., {'relations': [{'entity1': {'text': 'text1', 'entity_type': 'entity_type1'}, 'entity2': {'text': 'text2', 'entity_type': 'entity_type2'}, 'relation_type': 'relation_type1'}, ...]}
        
        The entity types must either be GeneOrGeneProduct (for a gene) or ChemicalEntity (for a drug/chemical), and the relation types must be either Positive_Correlation, Negative_Correlation, Association, or Bind. Guidelines for extracting relation types are given below.
        
        Positive_Correlation: Chemicals may cause higher gene expression or gene variants may trigger chemical adverse effects.

        Negative_Correlation: Chemicals may cause lower gene expression or gene variants may confer resistance to chemicals.

        Association: Non-specific associations and binding interactions between chemicals and gene promoters.'''

        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]
        
        return messages  
    
class MultiTemplateF:

    '''Separating relations by pairs of entity types. Chemical-Chemical Relations'''

    text = {
        "type": "string",
        "description": "Text of entity"
        }
    
    entity_type = {
        "type": "string",
        "description": "ChemicalEntity"
        }
    
    entity1 = {"type": "object",
               "description": "A biomedical entity that is a ChemicalEntity",
               "properties": {
                   "text": text,
                   "entity_type": entity_type
                   },
               "required": ["text", "entity_type"]
        }
    
    entity2 = deepcopy(entity1)
    
    relation_type = {
        "type": "string",
        "description": "A biomedical relationship that holds between two entities, either Positive_Correlation, Negative_Correlation, Association, Cotreatment, Drug_Interaction, or Conversion."
    }
    
    relation = {
        "type": "object",
        "description": "Two biomedical entities and a relationship that holds between them.",
        "properties": {
            "entity1": entity1,
            "entity2": entity2,
            "relation_type": relation_type
            },
        "required": ["entity1", "entity2", "relation_type"]
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
            relation = Relation({entity1, entity2}, el['relation_type'])
            relations.add(relation)

        return relations
    

    @classmethod
    def make_prompt(cls, example):
        
        system_content = '''You are a biomedical researcher interested in relationships holding between chemicals/drugs. When presented with a PubMed abstract, you find all chemical-chemical pairs of entities exhibiting one of a set of predefined relation types, and extract them as a list within a python dictionary, i.e., {'relations': [{'entity1': {'text': 'text1', 'entity_type': 'entity_type1'}, 'entity2': {'text': 'text2', 'entity_type': 'entity_type2'}, 'relation_type': 'relation_type1'}, ...]}
        
        The entity types must be ChemicalEntity (for a drug/chemical), and the relation types must be either Positive_Correlation, Negative_Correlation, Association, or Bind. Guidelines for extracting relation types are given below.
        
        Positive_Correlation: One chemical may increase the sensitivity or effectiveness of another.

        Negative_Correlation: One chemical may decrease the sensitivity or side effects of another.

        Association: Includes chemical conversions and non-specific associations.

        Drug_Interaction: Pharmacodynamic interactions between chemicals or drugs.

        Cotreatment: Combination therapy using multiple chemicals.

        Conversion: One chemical converting to another.'''

        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]
        
        return messages  
    
class MultiTemplateG:

    '''Separating relations by pairs of entity types. Chemical-Variant Relations'''

    text = {
        "type": "string",
        "description": "Text of entity"
        }
    
    entity_type = {
        "type": "string",
        "description": "ChemicalEntity or SequenceVariant"
        }
    
    entity1 = {"type": "object",
               "description": "A biomedical entity that is a ChemicalEntity or SequenceVariant",
               "properties": {
                   "text": text,
                   "entity_type": entity_type
                   },
               "required": ["text", "entity_type"]
        }
    
    entity2 = deepcopy(entity1)
    
    relation_type = {
        "type": "string",
        "description": "A biomedical relationship that holds between two entities, either Positive_Correlation, Negative_Correlation, or Association."
    }
    
    relation = {
        "type": "object",
        "description": "Two biomedical entities and a relationship that holds between them.",
        "properties": {
            "entity1": entity1,
            "entity2": entity2,
            "relation_type": relation_type
            },
        "required": ["entity1", "entity2", "relation_type"]
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
            relation = Relation({entity1, entity2}, el['relation_type'])
            relations.add(relation)

        return relations
    

    @classmethod
    def make_prompt(cls, example):
        
        system_content = '''You are a biomedical researcher interested in relationships holding between chemicals/drugs and gene variants. When presented with a PubMed abstract, you find all chemical-variant pairs of entities exhibiting one of a set of predefined relation types, and extract them as a list within a python dictionary, i.e., {'relations': [{'entity1': {'text': 'text1', 'entity_type': 'entity_type1'}, 'entity2': {'text': 'text2', 'entity_type': 'entity_type2'}, 'relation_type': 'relation_type1'}, ...]}
        
        The entity types must either be ChemicalEntity (for a drug/chemical) or SequenceVariant (for a gene variant), and the relation types must be either Positive_Correlation, Negative_Correlation, Association, or Bind. Guidelines for extracting relation types are given below.
        
        Positive_Correlation: Chemicals may cause higher expression of a gene variant or increase sensitivity due to a variant.

        Negative_Correlation: Chemicals may decrease gene expression due to a variant or the variant may confer resistance.

        Association: Captures relationships not defined as positive/negative correlations, like variants on chemical binding sites.'''

        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]
        
        return messages  
    
class Template6:
   
    @classmethod
    def extract_relations(cls, relation_list):
        relations = set()
        for el in relation_list:
            entity1 = Entity({el['entity1']['text']}, el['entity1']['entity_type'])
            entity2 = Entity({el['entity2']['text']}, el['entity2']['entity_type'])
            relation = Relation({entity1, entity2}, el['relation_type'])
            relations.add(relation)

        return relations

    @classmethod
    def make_prompt(cls, example):
        
        system_content = '''You are a biomedical researcher interested in relationships holding between genes, gene variants, chemicals/drugs, and diseases. When presented with a PubMed abstract, you find all pairs of entities belonging to a set of predefined relation types, and extract them as a list within a JSON, i.e., {'relations': [{'entity1': {'text': 'text1', 'entity_type': 'entity_type1'}, 'entity2': {'text': 'text2', 'entity_type': 'entity_type2'}, 'relation_type': 'relation_type1'}, ...]}
        
        The entity types must either be GeneOrGeneProduct (for a gene), SequenceVariant (for a gene variant), ChemicalEntity (for a chemical/drug), or DiseaseOrPhenotypicFeature (for a disease), and the relation types must be either Positive_Correlation, Negative_Correlation, Association, Bind, Cotreatment, Drug_Interaction, Comparison, or Conversion.  Relation types may only be valid for certain combinations of entity types and may have somewhat different meanings based on the pair of entity types.  Below are guidelines for extracting relations.
        
        Disease-Chemical Relations:

            Positive_Correlation: Chemicals may induce diseases, increase disease risk, or their levels may correlate with disease risk.

            Negative_Correlation: Chemicals used as drugs may treat diseases or decrease disease susceptibility.

            Association: Captures relationships not clearly defined as positive or negative correlations, such as drugs with potential safety concerns.

        Disease-Gene Relations:

            Positive_Correlation: Overexpression or side effects of proteins (from genes) may cause diseases.

            Negative_Correlation: Proteins used as drugs may treat diseases or their absence may cause diseases.

            Association: Includes functional genes preventing diseases and other associations not falling under positive/negative correlations.

        Disease-Variant Relations:

            Positive_Correlation: Variants may increase disease risk, contribute to disease susceptibility, or cause protein deficiencies leading to diseases.

            Negative_Correlation: Variants may decrease disease risk.

            Association: Includes variants associated with disease prevalence and those that cannot be categorized as causing the disease.

        Gene-Gene Relations:

            Positive_Correlation: Genes may show positive correlations in expression or regulatory functions.

            Negative_Correlation: Genes may show negative correlations in expression or regulatory functions.
            
            Association: Modifications like phosphorylation or other associations that cannot be categorized differently.
            
            Bind: Physical interactions between proteins, including protein binding at gene promoters.

        Gene-Chemical Relations:

            Positive_Correlation: Chemicals may cause higher gene expression or gene variants may trigger chemical adverse effects.

            Negative_Correlation: Chemicals may cause lower gene expression or gene variants may confer resistance to chemicals.

            Association: Non-specific associations and binding interactions between chemicals and gene promoters.

        Chemical-Chemical Relations:

            Positive_Correlation: One chemical may increase the sensitivity or effectiveness of another.

            Negative_Correlation: One chemical may decrease the sensitivity or side effects of another.

            Association: Includes chemical conversions and non-specific associations.
            
            Drug_Interaction: Pharmacodynamic interactions between chemicals or drugs.

            Cotreatment: Combination therapy using multiple chemicals.
            
            Conversion: One chemical converting to another.

        Chemical-Variant Relations:

            Positive_Correlation: Chemicals may cause higher expression of a gene variant or increase sensitivity due to a variant.

            Negative_Correlation: Chemicals may decrease gene expression due to a variant or the variant may confer resistance.

            Association: Captures relationships not defined as positive/negative correlations, like variants on chemical binding sites.'''
                
        user_content = f'Title: {example.title}\n\n Abstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]
        
        return messages  

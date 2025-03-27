
from util.Concept import *
from util.Ling import *
from util.Terminology import *
from util.Util import *
from Sieve import *

class StemmingSieve(Sieve):
    def __init__(self):
        super(StemmingSieve, self).__init__()
        pass
    
    def apply(self, concept):
        self.transformName(concept)
        return StemmingSieve.normalize(concept)
    
    def transformName(self, concept):
        namesForTransformation = concept.getNamesKnowledgeBase()
        transformedNames = []      
        
        for nameForTransformation in namesForTransformation:
            transformedNames = Util.setList(transformedNames, Ling.getStemmedPhrase(nameForTransformation)) 
        concept.setStemmedNamesKnowledgeBase(transformedNames)  

    def normalize(self, concept):
        for name in concept.getStemmedNamesKnowledgeBase():
            cui = StemmingSieve.exactMatchSieve(name)       
            if not cui.equals(""):
                return cui
        return ""  
    
    def exactMatchSieve(self, name):
        cui = ""
        # checks against names normalized by multi-pass sieve
        cui = getTerminologyNameCui(Terminology.getStemmedNormalizedNameToCuiListMap(), name)
        if not cui == "":
            return cui
        
        # checks against names in training data
        cui = getTerminologyNameCui(Sieve.getTrainingDataTerminology().getStemmedNameToCuiListMap(), name)
        if not cui == "":
            return cui       
        
        # checks against names in dictionary
        cui = getTerminologyNameCui(Sieve.getStandardTerminology().getStemmedNameToCuiListMap(), name);       
        return cui

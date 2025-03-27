from Ling import *
from Abbreviation import *
from Util import  *
class Concept():
    def __init__(self, indexes, name, goldMeSHorSNOMEDCui, goldOMIMCuis):
        self.indexes = indexes
        self.name = Ling.correctSpelling(name.toLowerCase().strip())
        self.goldMeSHorSNOMEDCui = goldMeSHorSNOMEDCui
        self.goldOMIMCuis = goldOMIMCuis
        self.stemmedName =  None
        self.goldMeSHorSNOMEDCui =  None
        self.goldOMIMCuis =  None
        self.cui  = None
        self.alternateCuis =  None
        self.normalizingSieveLevel = 0
        self.namesKnowledgeBase = []
        self.stemmedNamesKnowledgeBase = []
    

    def setIndexes(self, indexes):
        self.indexes = indexes
    
    def getIndexes(self):
        return self.indexes
    
    def setName(self, name):
        self.name = Ling.correctSpelling(name.lower().strip())
    
    def getName(self):
        return self.name
    

    def setNameExpansion(self, text, abbreviationObject):
        self.nameExpansion = Abbreviation.getAbbreviationExpansion(abbreviationObject, text, self.name, self.indexes)   

    
    def getNameExpansion(self):
        return self.nameExpansion

    
    def setStemmedName(self):
        stemmedName = Ling.getStemmedPhrase(self.name)
    
    def getStemmedName(self):
        return self.stemmedName
    
    
    def setCui(self, cui):
        self.cui = cui
        
    def getCui(self):
        return self.cui
    
    def setAlternateCuis(self, alternateCuis):
        self.alternateCuis = []
        for  alternateCui in alternateCuis:
             alternateCuis = Util.setList(self.alternateCuis, alternateCui)
    
    def getAlternateCuis(self):
        return self.alternateCuis
    
    def setNormalizingSieveLevel(self, sieveLevel):
        self.normalizingSieveLevel = sieveLevel;
    
    
    def getNormalizingSieve(self):
        return self.normalizingSieveLevel
    
    def getGoldMeSHorSNOMEDCui(self):
        return self.goldMeSHorSNOMEDCui   
    
    def getGoldOMIMCuis(self):
        return self.goldOMIMCuis
    
    
    def getGoldCui(self):
        if not self.goldMeSHorSNOMEDCui=="":
            return self.goldMeSHorSNOMEDCui
        else:
            return self.goldOMIMCuis.get(0) if len(self.goldOMIMCuis) == 1 else str(goldOMIMCuis)
    
 
    def reinitializeNamesKnowledgeBase(self):
        self.namesKnowledgeBase = []
    
    def setNamesKnowledgeBase(self, name):
        self.namesKnowledgeBase = Util.setList(self.namesKnowledgeBase, name)
    
    
    def setNamesKnowledgeBase(self,namesList):
        self.namesKnowledgeBase = Util.addUnique(self.namesKnowledgeBase, namesList)
    
    
    def  getNamesKnowledgeBase(self):
        return self.namesKnowledgeBase
    

    def setStemmedNamesKnowledgeBase(self, namesList):
        self.stemmedNamesKnowledgeBase = Util.addUnique(self.stemmedNamesKnowledgeBase, namesList)
    
    def getStemmedNamesKnowledgeBase(self):
        return self.stemmedNamesKnowledgeBase

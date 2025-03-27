import Main;
import MultiPassSieveNormalizer;
import util.AmbiguityResolution;
import util.Concept;
import util.Terminology;
import util.Util;

class Sieve():
    def __init__(self):
         pass
    
    standardTerminology = Terminology() 

    def setStandardTerminology(self):
        self.standardTerminology.loadTerminology()
    
    def getStandardTerminology(self):
        return self.standardTerminology
    
    trainingDataTerminology = Terminology()
    
    def setTrainingDataTerminology(self):
        self.trainingDataTerminology.loadTrainingDataTerminology(Main.training_data_dir)
       
    def getTrainingDataTerminology(self):
        return self.trainingDataTerminology
    
    def getTerminologyNameCuis(nameToCuiListMap, name):
        return nameToCuiListMap.get(name) if nameToCuiListMap.__contains__(name) else None
    
    
    def getTerminologyNameCui(nameToCuiListMap, name):
        return nameToCuiListMap.get(name).get(0) if nameToCuiListMap.containsKey(name) and len(nameToCuiListMap.get(name)) == 1 else  ""
    
    def exactMatchSieve(self,name):
        cui = ""
        cui = self.getTerminologyNameCui(Terminology.getNormalizedNameToCuiListMap(), name)
        if not cui=="":
            return cui
        
        cui = self.getTerminologyNameCui(self.trainingDataTerminology.getNameToCuiListMap(), name)
        if not cui == "": 
            return cui;       
        
        return self.getTerminologyNameCui(self.standardTerminology.getNameToCuiListMap(), name)           
    

    def getAlternateCuis(self, cui):
        alternateCuis = []
        if self.trainingDataTerminology.getCuiAlternateCuiMap().__contains__(cui):
            alternateCuis.addAll(trainingDataTerminology.getCuiAlternateCuiMap()[cui])
        
        if self.standardTerminology.getCuiAlternateCuiMap().__contains__(cui):
            alternateCuis.addAll(self.standardTerminology.getCuiAlternateCuiMap().get(cui))
        return alternateCuis
 
    def normalize(namesKnowledgeBase):
        for name in namesKnowledgeBase:
            cui = self.exactMatchSieve(name)       
            if  not cui == "":
                return cui
        return ""
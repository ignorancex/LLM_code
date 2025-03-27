
import tool.Main;
from sieves.CompoundPhraseSieve import *
from sieves.SimpleNameSieve import *
import os

class Terminology ():
    def __init__(self):
            
        self.tokenToNameListMap = {}
        self.nameToCuiListMap = {}
        self.simpleNameToCuiListMap = {}
        self.compoundNameToCuiListMap = {}
        self.cuiToNameListMap = {}
        self.stemmedNameToCuiListMap = {}
        self.cuiToStemmedNameListMap = {}
        self.cuiAlternateCuiMap = {}
        self.cuiNameFileListMap = {}
    
    def getTokenToNameListMap(self):
        return self.tokenToNameListMap
    
    
    def getNameToCuiListMap(self):
        return self.nameToCuiListMap

    
    def getSimpleNameToCuiListMap(self):
        return self.simpleNameToCuiListMap
    
    
    def getCompoundNameToCuiListMap(self):
        return self.compoundNameToCuiListMap

    
    def setCompoundNameToCuiListMap(self, name,  cui):
        self.compoundNameToCuiListMap = Util.setMap(self.compoundNameToCuiListMap, name, cui)
    
    def  getCuiToNameListMap(self):
        return self.cuiToNameListMap
    
    def getStemmedNameToCuiListMap(self):
        return self.stemmedNameToCuiListMap
    
    def getCuiToStemmedNameListMap(self):
        return self.cuiToStemmedNameListMap
    
    def getCuiAlternateCuiMap(self):
        return self.cuiAlternateCuiMap
    
    def get_preferredID_set_altID(self, identifiers):
        preferredID = ""
        set_ = False
        altIDs = []
        
        for i in range(0, len(identifiers)):
            if "OMIM" in identifiers[i]:
                identifiers[i] = identifiers[i].split(":")[1]
            if i == 0:
                preferredID = identifiers[i]

            if identifiers[i][0].isalpha() and set_ == False:
                preferredID = identifiers[i]                     
                set_ = True
                continue
                       
            altIDs.append(identifiers[i])
        
        
        if len(altIDs)!=0:
            self.cuiAlternateCuiMap[preferredID]=altIDs
               
        return preferredID;   

    def loadMap(self, conceptName, cui):   
        self.nameToCuiListMap = Util.setMap(self.nameToCuiListMap, conceptName, cui);
        self.cuiToNameListMap = Util.setMap(self.cuiToNameListMap, cui, conceptName);

        self.stemmedConceptName = Ling.getStemmedPhrase(conceptName)
        self.stemmedNameToCuiListMap = Util.setMap(self.stemmedNameToCuiListMap, self.stemmedConceptName, cui)
        self.cuiToStemmedNameListMap = Util.setMap(self.cuiToStemmedNameListMap, cui, self.stemmedConceptName)
        
        conceptNameTokens = conceptName.split("\\s")
        for conceptNameToken in conceptNameTokens):
            if conceptNameToken in Ling.getStopwordsList():
                continue
            self.tokenToNameListMap = Util.setMap(self.tokenToNameListMap, conceptNameToken, conceptName)
    
        
        if "semeval" in Main.training_data_dir.toString():
            CompoundPhraseSieve.setCompoundNameTerminology(self, conceptName, conceptNameTokens, cui)       
        
        elif "|" in cui:
            self.nameToCuiListMap.remove(conceptName)
            self.stemmedNameToCuiListMap.remove(stemmedConceptName)
            for conceptNameToken in conceptNameTokens:
                if conceptNameToken in Ling.getStopwordsList():
                    continue;       
                self.tokenToNameListMap.get(conceptNameToken).remove(conceptName)
            
            self.compoundNameToCuiListMap = Util.setMap(self.compoundNameToCuiListMap, conceptName, cui)
  
    
    def loadTerminology(self, file_name = os.path.join()):
        cui = ""
        try (BufferedReader br = new BufferedReader(new FileReader(terminologyFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.equals(""))
                    continue;
                String[] token = line.split("\\|\\|");
                
                cui = token[0].contains("|") ? get_preferredID_set_altID(token[0].split("\\|")) : token[0];
                
                String[] conceptNames = token[1].toLowerCase().split("\\|");
                
                for (String conceptName : conceptNames)
                    loadMaps(conceptName, cui);
            }
        }
    }        
    
    private void setOMIM(String cuis, String MeSHorSNOMEDcuis, String conceptName) {
        if (MeSHorSNOMEDcuis.equals("")) {
            cuis = cuis.replaceAll("OMIM:", "");
            loadMaps(conceptName, cuis);
        }
        else {
            String[] cuis_arr = cuis.split("\\|");
            for (String cui : cuis_arr) {
                if (!cui.contains("OMIM"))
                    continue;
                cui = cui.split(":")[1];
                cuiAlternateCuiMap = Util.setMap(cuiAlternateCuiMap, MeSHorSNOMEDcuis, cui);
            }
        }
    }
    
    public static List<String> getOMIMCuis(String[] cuis) {
        List<String> OMIMcuis = new ArrayList<>();
        for (String cui : cuis) {
            if (!cui.contains("OMIM"))
                continue;
            cui = cui.split(":")[1];
            OMIMcuis = Util.setList(OMIMcuis, cui);
        }
        return OMIMcuis;
    }
    
    public static String getMeSHorSNOMEDCuis(String[] cuis) {
        String cuiStr = "";
        for (String cui : cuis) {
            if (cui.contains("OMIM"))
                continue;
            cuiStr = cuiStr.equals("") ? cui : cuiStr+"|"+cui;
        }
        return cuiStr;
    }
    
    
    
    private static Map<String, List<String>> normalizedNameToCuiListMap = new HashMap<>();
    public static void setNormalizedNameToCuiListMap(String name, String cui) {
        normalizedNameToCuiListMap = Util.setMap(normalizedNameToCuiListMap, name, cui);
    }
    public static Map<String, List<String>> getNormalizedNameToCuiListMap() {
        return normalizedNameToCuiListMap;
    }
    
    private static Map<String, List<String>> stemmedNormalizedNameToCuiListMap = new HashMap<>();
    public static void setStemmedNormalizedNameToCuiListMap(String stemmedName, String cui) {
        stemmedNormalizedNameToCuiListMap = Util.setMap(stemmedNormalizedNameToCuiListMap, stemmedName, cui);
    }
    public static Map<String, List<String>> getStemmedNormalizedNameToCuiListMap() {
        return stemmedNormalizedNameToCuiListMap;
    }        
    
    public static void storeNormalizedConcept(Concept concept) {
        setNormalizedNameToCuiListMap(concept.getNormalizingSieve() == 2 ? concept.getNameExpansion() : concept.getName(), concept.getCui());
        setStemmedNormalizedNameToCuiListMap(concept.getNormalizingSieve() == 2 ? Ling.getStemmedPhrase(concept.getNameExpansion()) : concept.getStemmedName(), concept.getCui());
    }
    
}
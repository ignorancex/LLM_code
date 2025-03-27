
import util.Concept
import util.Ling
import util.Util

class AffixationSieve(Sieve):
    def __init__(self):
        super().__init__()
    
    def apply(self, concept):
        self.transformName(concept)
        return normalize(concept.getNamesKnowledgeBase())  
    
    def transformName(self, concept):
        namesForTransformation = concept.getNamesKnowledgeBase()
        transformedNames = []
        
        for nameForTransformation in namesForTransformation:
            transformedNames = Util.addUnique(transformedNames, self.affix(nameForTransformation))
        
        concept.setNamesKnowledgeBase(transformedNames)       
    
    def getAllStringTokenSuffixationCombinations(self, stringTokens):
        suffixatedPhrases = []
        for stringToken in stringTokens:            
            suffix = self.ling.getSuffixStr(stringToken); 
            forSuffixation = None if suffix=="" else self.ling.getSuffixMap()[suffix]
                        
            if len(suffixatedPhrases) == 0:
                if forSuffixation == None:
                    suffixatedPhrases.append(stringToken)
                elif len(forSuffixation) == 0:
                    suffixatedPhrases.append(stringToken.replace(suffix, ""))
                else:
                    for i in range(forSuffixation.size()):
                        suffixatedPhrases.append(stringToken.replace(suffix, forSuffixation[i]))
                
            
            else:
                if (forSuffixation == None):
                    for i in range(len(suffixatedPhrases)): 
                        suffixatedPhrases[i] = suffixatedPhrases[i] + " " +stringToken
                
                elif (len(forSuffixation) == 0):
                    for  i in range(suffixatedPhrases.size()):
                        suffixatedPhrases[i]  = suffixatedPhrases[i] + " " + stringToken.replace(suffix, "")  
                
                else:
                    tempSuffixatedPhrases = []
                    for i in range(suffixatedPhrases.size()):
                        suffixatedPhrase = suffixatedPhrases[i]
                        for j in range(forSuffixation.size()):
                            tempSuffixatedPhrases.append(suffixatedPhrase+" "+stringToken.replace(suffix, forSuffixation.get(j)))
                    
                    suffixatedPhrases = list(tempSuffixatedPhrases)
                    tempSuffixatedPhrases = None
  
        return suffixatedPhrases

    def getUniformStringTokenSuffixations(self, stringTokens, string):
        suffixatedPhrases = []
        for stringToken in stringTokens:            
            suffix = self.ling.getSuffixStr(stringToken); 
            forSuffixation = None if suffix=="" else self.ling.getSuffixMap()[suffix]
            if forSuffixation == None:
                continue
            if (len(forSuffixation) == 0):
                suffixatedPhrases.append(string.replace(suffix, ""))
                continue

            for i in range(forSuffixation.size()):
                suffixatedPhrases.append(string.replace(suffix, forSuffixation[i]))
        return suffixatedPhrases

    def suffixation(self, stringTokens, string):
        suffixatedPhrases = self.get_suffixation_combinations(stringTokens)
        suffixatedPhrases.extend(self.getUniformStringTokenSuffixations(stringTokens, string))
        return suffixatedPhrases

    def prefixation(self, stringTokens, string):
        prefixatedPhrase = ""
        for stringToken in  stringTokens:
            prefix = self.ling.getPrefixStr(stringToken)
            forPrefixation = "" if prefix=="" else self.ling.getPrefixMap()[prefix]
            prefixatedPhrase = (stringToken if prefix=="" else stringToken.replace(prefix, forPrefixation)) if prefixatedPhrase=="" else (prefixatedPhrase + " " + stringToken if prefix=="" else prefixatedPhrase + " " + stringToken.replace(prefix, forPrefixation))
        return prefixatedPhrase  


    def affixation(self, stringTokens, string):
        affixatedPhrase = ""
        for stringToken in stringTokens:
            
            affix = (self.ling.AFFIX.split("|")[0] if  self.ling.AFFIX.split("|")[0] in stringToken else self.ling.AFFIX.split("|")[1]) if search(".*("+self.ling.AFFIX+").*", stringToken) else ""
            
            forAffixation = "" if affix=="" else self.ling.getAffixMap()[affix]

            affixatedPhrase = (stringToken if affix=="" else stringToken.replace(affix, forAffixation)) if affixatedPhrase=="" else (affixatedPhrase+" "+stringToken if affix=="" else affixatedPhrase + " " + stringToken.replace(affix, forAffixation))
        return affixatedPhrase


    def affix(self,string):
        stringTokens = string.split(" ")
        newPhrases = self.suffixation(stringTokens, string)
        newPhrases = Util.setList(newPhrases, self.prefixation(stringTokens, string))
        newPhrases = Util.setList(newPhrases, self.affixation(stringTokens, string))       
        return newPhrases
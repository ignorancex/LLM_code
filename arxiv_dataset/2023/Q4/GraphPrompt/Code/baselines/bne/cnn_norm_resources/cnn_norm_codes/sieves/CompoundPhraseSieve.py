import util.Ling
import util.Terminology
import util.Util

class CompoundPhraseSieve(Sieve):
    def __init__(self):
        super().__init__()
    
    def applyNCBI(self,name):
        self.cui = self.apply(name)
        if not self.cui.equals("") or  (not " and " in name  and  not " or " in name):
            return self.cui
        
        compoundWord = "and" if " and " in name else "or"
        nameTokens = name.split(" ")
        index = Util.getTokenIndex(nameTokens, compoundWord)
        
        if index == 1:
            replacement1 = nameTokens[0]
            replacement2 = nameTokens[2]+" "+nameTokens[3] if nameTokens[2] == "the" else nameTokens[2]

            phrase = replacement1+" "+compoundWord+" "+replacement2

            replacement2 = nameTokens[3] if nameTokens[2]=="the" else nameTokens[2]

            cui1 = exactMatchSieve(name.replace(phrase, replacement1))
                        
            cui2 = exactMatchSieve(name.replace(phrase, replacement2))
            if not cui1 == "" and not cui2 == "":
                return cui2+"|"+cui1 if Sieve.getTrainingDataTerminology().getCuiToNameListMap().__contains__(cui2+"|"+cui1) else cui1+"|"+cui2
     
        return ""
    
    def apply(self, name):
        cui = getTerminologyNameCui(Sieve.getTrainingDataTerminology().getCompoundNameToCuiListMap(), name)
        if not cui == "":
            return cui
        
        return getTerminologyNameCui(Sieve.getStandardTerminology().getCompoundNameToCuiListMap(), name);   

    
    def setCompoundNameTerminology( terminology, conceptName, conceptNameTokens, cui):
        if "and/or" in conceptName:
            indexes = Util.getTokenIndexes(conceptNameTokens, "and/or")
            if len(indexes) == 1:
                index = indexes[0]     
                if search("[a-zA-Z]+, [a-zA-Z]+ and/or [a-zA-Z]+.*", conceptName):   
                    replacement1 = conceptNameTokens[index-2].replace(",", "");
                    replacement2 = conceptNameTokens[index-1]
                    replacement3 = conceptNameTokens[index+1]
                    phrase = replacement1+", "+replacement2+" "+conceptNameTokens[index]+" "+replacement3;        
                    
                    terminology.setCompoundNameToCuiListMap(conceptName.replace(phrase, replacement1), cui)
                    terminology.setCompoundNameToCuiListMap(conceptName.replace(phrase, replacement2), cui)
                    terminology.setCompoundNameToCuiListMap(conceptName.replace(phrase, replacement3), cui)
                
                else:
                    replacement1 = conceptNameTokens[index-1]
                    replacement2 = conceptNameTokens[index+1]+" "+conceptNameTokens[index+2] if len(conceptNameTokens) -1 == index+2 else conceptNameTokens[index+1]
                    phrase = replacement1+" "+conceptNameTokens[index]+" "+replacement2;        
                    terminology.setCompoundNameToCuiListMap(conceptName.replace(phrase, replacement1), cui)
                    terminology.setCompoundNameToCuiListMap(conceptName.replace(phrase, replacement2), cui)
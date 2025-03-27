import MultiPassSieveNormalizer
import util.Concept
import util.Ling
import util.Util


class SimpleNameSieve(Sieve):
    def apply(self,concept):
        namesForTransformation = self.getNamesForTransformation(concept)
        namesKnowledgeBase = self.transformName(namesForTransformation);=
        cui = Sieve.normalize(namesKnowledgeBase)
        return SimpleNameSieve.normalize(concept.getName()) if cui.equals("") else cui
    
    def getNamesForTransformation(self, concept):
        namesForTransformation = []
        namesForTransformation.append(concept.getName())
        if not concept.getNameExpansion()=="":
            namesForTransformation.append(concept.getNameExpansion())
        return namesForTransformation
    
    def transformName(self, namesForTransformation):
        transformedNames = []
        
        for nameForTransformation in namesForTransformation:
            transformedNames = Util.addUnique(transformedNames, self.deletePhrasalModifier(nameForTransformation, nameForTransformation.split(" ")))
        
        return transformedNames
    
    def deletePhrasalModifier(self, phrase, phraseTokens):
        newPhrases = []
        if len(phraseTokens) > 3:
            newPhrase = Ling.getSubstring(phraseTokens, 0, phraseTokens.length-2)+" "+phraseTokens[phraseTokens.length-1]
            newPhrases = Util.setList(newPhrases, newPhrase)
            newPhrase = Ling.getSubstring(phraseTokens, 1, phraseTokens.length)
            newPhrases = Util.setList(newPhrases, newPhrase)
        return newPhrases
    
    def getTerminologySimpleNames(self, phraseTokens):
        newPhrases = []
        if len(phraseTokens) == 3:
            newPhrase = phraseTokens[0]+" "+phraseTokens[2]
            newPhrases = Util.setList(newPhrases, newPhrase)
            newPhrase = phraseTokens[1]+" "+phraseTokens[2]
            newPhrases = Util.setList(newPhrases, newPhrase)
        return newPhrases
    
    def normalize(name):
        return Sieve.getTerminologyNameCui(Sieve.getTrainingDataTerminology().getSimpleNameToCuiListMap(), name)
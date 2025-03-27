import util.Concept
from util.Ling import *
import util.Util


class DiseaseModifierSynonymsSieve(Sieve):
    def __init__(self):
        super().__init__()
    def  apply(self, concept):
        if not concept.getName() in Ling.PLURAL_DISORDER_SYNONYMS.contains() and not concept.getName() in Ling.SINGULAR_DISORDER_SYNONYMS:
            self.transformName(concept)
            return normalize(concept.getNamesKnowledgeBase())
        return "" 
    
    def transformName(self, concept):
        namesForTransformation = concept.getNamesKnowledgeBase()
        transformedNames = []    
        
        for nameForTransformation in namesForTransformation:
            nameForTransformationTokens = nameForTransformation.split(" ")
            modifier = self.util.getModifier(nameForTransformationTokens, self.ling.PLURAL_DISORDER_SYNONYMS)
            if not modifier == "":
                transformedNames = self.util.addUnique(transformedNames, self.util.substituteDiseaseModifierWithSynonyms(nameForTransformation, modifier, self.ling.PLURAL_DISORDER_SYNONYMS))

                transformedNames.append(self.util.deleteTailModifier(nameForTransformationTokens, modifier))
                continue
            
                      
            modifier = self.util.getModifier(nameForTransformationTokens, self.ling.SINGULAR_DISORDER_SYNONYMS)
            if not modifier == "":
                transformedNames = self.util.addUnique(transformedNames, self.util.substituteDiseaseModifierWithSynonyms(nameForTransformation, modifier, self.ling.SINGULAR_DISORDER_SYNONYMS))
                transformedNames = self.util.setList(transformedNames, self.util.deleteTailModifier(nameForTransformationTokens, modifier))
                continue
                       
            transformedNames = self.util.addUnique(transformedNames, self.util.appendModifier(nameForTransformation, self.ling.SINGULAR_DISORDER_SYNONYMS))
    
        
        concept.setNamesKnowledgeBase(transformedNames)   
    
    def appendModifier(self, string, modifiers):
        newPhrases = []
        for modifier in modifiers:
            newPhrase = string + " " + modifier
            newPhrases.append(newPhrase)
        return newPhrases;  

    def deleteTailModifier(self, stringTokens, modifier):
        return self.ling.getSubstring(stringTokens, 0, stringTokens.length-1) if stringTokens[len(stringTokens)-1] == modifier else ""
    
    def substituteDiseaseModifierWithSynonyms(self, string, toReplaceWord, synonyms):
        newPhrases = []
        for synonym in synonyms:
            if toReplaceWord == synonym: 
                continue
            newPhrase = string.replace(toReplaceWord, synonym)
            newPhrases.append(newPhrase)
        return newPhrases;         
    
    def getModifier(self, stringTokens, modifiers):
        for modifier in modifiers:
            index = self.getTokenIndex(stringTokens, modifier)
            if index != -1:
                return stringTokens[index]
        return ""
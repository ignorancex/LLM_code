
import util.Concept
import util.Ling
import util.Util

class SymbolReplacementSieve(Sieve):
    def __init__(self):
        pass
    
    def apply(self, concept):
        self.transformName(concept)
        return normalize(concept.getNamesKnowledgeBase())   
    
    def transformName(self, concept):
        namesForTransformation = concept.getNamesKnowledgeBase()
        transformedNames = []      
        
        for nameForTransformation in namesForTransformation:
            transformedNames = Util.addUnique(transformedNames, self.substituteSymbolsInStringWithWords(nameForTransformation))
            transformedNames = Util.addUnique(transformedNames, self.substituteWordsInStringWithSymbols(nameForTransformation))
        
        concept.setNamesKnowledgeBase(transformedNames)
    
    def getClinicalReportTypeSubstitutions(string):
        newStrings = []
        for digit in Ling.getDigitToWordMap().keys():
            if not digit in string:
                continue;
            wordsList = Ling.getDigitToWordMap()[digit]
            for word in wordsList:
                newString = string.replaceAll(digit, word)
                if not newString ==  string:
                    newStrings = Util.setList(newStrings, newString)
        return newStrings

    def getBiomedicalTypeSubstitutions(string):
        if "and/or" in string:
            string = string.replaceAll("and/or", "and")
        if "/" in string:
            string = string.replaceAll("/", " and ")
        if " (" in string and ")" in string:
            string = string.replace(" (", "").replace(")", "")
        elif "(" in string and ")" in string:
            string = string.replace("(", "").replace(")", "")
        return string
    
    def substituteSymbolsInStringWithWords(self, string):
        newStrings = self.getClinicalReportTypeSubstitutions(string)
        tempNewStrings = []
        for newString in newStrings:
            tempNewStrings = Util.setList(tempNewStrings, self.getBiomedicalTypeSubstitutions(newString))       
        newStrings = Util.addUnique(newStrings, tempNewStrings)
        newStrings = Util.setList(newStrings, self.getBiomedicalTypeSubstitutions(string))      
        return newStrings
    
    def substituteWordsInStringWithSymbols(self, string):
        newStrings = []
        for word in Ling.getWordToDigitMap().keys():
            if not word in string:
                continue
            digit = Ling.getWordToDigitMap()[word]
            newString = string.replaceAll(word, digit)
            if not newString ==  string:
                newStrings = Util.setList(newStrings, newString)
        return newStrings
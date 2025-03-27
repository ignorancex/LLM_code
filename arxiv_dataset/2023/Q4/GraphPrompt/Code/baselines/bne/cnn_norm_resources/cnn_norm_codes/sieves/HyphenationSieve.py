
import java.util.ArrayList;
import java.util.List;
import tool.util.Concept;
import tool.util.Util;


class HyphenationSieve(Sieve):
    def __init__(self):
        super(HyphenationSieve, self).__init__()
    
    def apply(self, concept):
        self.transformName(concept)
        return normalize(concept.getNamesKnowledgeBase())     
 
    def transformName(self, concept):
        namesForTransformation = concept.getNamesKnowledgeBase()
        transformedNames = []     
        
        for nameForTransformation in namesForTransformation:
            transformedNames = Util.addUnique(transformedNames, self.hyphenateString(nameForTransformation.split(" ")))
            transformedNames = Util.addUnique(transformedNames, self.dehyphenateString(nameForTransformation.split("-")))

        concept.setNamesKnowledgeBase(transformedNames) 
    
    
    def hyphenateString(stringTokens):
        hyphenatedStrings = []
        for i in range(1,len(stringTokens)):
            hyphenatedString = ""
            for j in range(0,len(stringTokens)):
                if j == i:
                    hyphenatedString += "-"+stringTokens[j]
                else:
                    hyphenatedString = stringTokens[j] if hyphenatedString.equals("") else hyphenatedString+" "+stringTokens[j]

            hyphenatedStrings = Util.setList(hyphenatedStrings, hyphenatedString)
        
        return hyphenatedStrings
   

    def dehyphenateString(stringTokens):
        dehyphenatedStrings = []
        for i in range(1,len(stringTokens)):
            dehyphenatedString = ""
            for j in range(len(stringTokens)):
                if j == i:
                    dehyphenatedString += " "+stringTokens[j]
                else:
                    dehyphenatedString = stringTokens[j]if dehyphenatedString=="" else dehyphenatedString+"-"+stringTokens[j]
            
            dehyphenatedStrings = Util.setList(dehyphenatedStrings, dehyphenatedString)
        return dehyphenatedStrings
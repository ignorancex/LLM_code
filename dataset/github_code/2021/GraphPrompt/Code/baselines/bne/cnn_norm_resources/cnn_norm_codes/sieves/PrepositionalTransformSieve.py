import util.Concept;
import util.Ling;
import util.Util;

class PrepositionalTransformSieve(Sieve):
    def __init__(self):
        super().__init__()
        pass

    
    def apply(self, concept):
        PrepositionalTransformSieve.init(concept)
        self.transformName(concept)
        return normalize(concept.getNamesKnowledgeBase())
    
    def init(concept):
        concept.setNamesKnowledgeBase(concept.getName())
        if not concept.getNameExpansion() == "":
            concept.setNamesKnowledgeBase(concept.getNameExpansion())
    
    def transformName(self, concept):
        namesForTransformation = concept.getNamesKnowledgeBase()
        transformedNames = []
        
        for nameForTransformation in namesForTransformation:
            prepositionInName = Ling.getStringPreposition(nameForTransformation)           
            
            if not prepositionInName == "":
                transformedNames = Util.addUnique(transformedNames, self.substitutePrepositionsInPhrase(prepositionInName, nameForTransformation));
                transformedNames = Util.setList(transformedNames, self.swapPhrasalSubjectAndObject(prepositionInName, nameForTransformation.split(" ")))
            
            else:
                transformedNames = Util.addUnique(transformedNames, self.insertPrepositionsInPhrase(nameForTransformation, nameForTransformation.split(" ")))

        concept.setNamesKnowledgeBase(transformedNames)
    
    def insertPrepositionsInPhrase(phrase,  phraseTokens):
        newPrepositionalPhrases = []
        for preposition in Ling.PREPOSITIONS:
            newPrepositionalPhrase = (Ling.getSubstring(phraseTokens, 1, phraseTokens.length)+" "+preposition+" "+phraseTokens[0]).strip()
            newPrepositionalPhrases = Util.setList(newPrepositionalPhrases, newPrepositionalPhrase)

            newPrepositionalPhrase = (phraseTokens[phraseTokens.length-1]+" "+preposition+" "+Ling.getSubstring(phraseTokens, 0, phraseTokens.length-1)).strip()
            newPrepositionalPhrases = Util.setList(newPrepositionalPhrases, newPrepositionalPhrase)
        return newPrepositionalPhrases
    
    def substitutePrepositionsInPhrase(prepositionInPhrase, phrase):
        newPrepositionalPhrases = []
        for preposition in Ling.PREPOSITIONS:
            if preposition == prepositionInPhrase:
                continue

            newPrepositionalPhrase = (phrase.replace(" "+prepositionInPhrase+" ", " "+preposition+" ")).strip()
            newPrepositionalPhrases = Util.setList(newPrepositionalPhrases, newPrepositionalPhrase)
        return newPrepositionalPhrases
    
    def swapPhrasalSubjectAndObject(prepositionInPhrase, phraseTokens):
        prepositionTokenIndex = Util.getTokenIndex(phraseTokens, prepositionInPhrase)
        return (Ling.getSubstring(phraseTokens, prepositionTokenIndex+1, phraseTokens.length)+" "+ Ling.getSubstring(phraseTokens, 0, prepositionTokenIndex)).strip() if prepositionTokenIndex !=-1 else ""

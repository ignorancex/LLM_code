
import util.Concept;
import util.Ling;
import util.Terminology;
import util.Util;

class PartialMatchSieve(Seive):
    def _init(self):
        pass    
    
    def apply(self, concept):
        
        name = concept.getNameExpansion() if concept.getNameExpansion() == "" else  concept.getName()
        nameTokens = name.split(" ")
        cuiList = self.partialMatch(name, nameTokens);        
        return cuiList[0] if len(cuiList) == 1 else ""
    
    def partialMatch(self, phrase, phraseTokens):
        cuiList = []
        prevPartialMatchedPhrases = []
                
        for phraseToken in  phraseTokens:
            if phraseToken in Ling.getStopwordsList():
                continue
            candidatePhrases = None
            map = -1
            
            if Sieve.getTrainingDataTerminology().getTokenToNameListMap().__contains__(phraseToken):
                candidatePhrases = Sieve.getTrainingDataTerminology().getTokenToNameListMap()[phraseToken]
                map = 2
            
            elif Sieve.getStandardTerminology().getTokenToNameListMap().__contains__(phraseToken):
                candidatePhrases = Sieve.getStandardTerminology().getTokenToNameListMap()[phraseToken]
                map = 3
            
            if candidatePhrases == None:
                continue
                        
            candidatePhrases.removeAll(prevPartialMatchedPhrases)
            
            if map == 2 and candidatePhrases.isEmpty() and Sieve.getStandardTerminology().getTokenToNameListMap().__contains__(phraseToken):
                candidatePhrases = Sieve.getStandardTerminology().getTokenToNameListMap()[phraseToken]
                map = 3;           
            
            cuiList = Sieve.getTrainingDataTerminology() if self.exactTokenMatchCondition(phrase, candidatePhrases, map == 2 else Sieve.getStandardTerminology(), cuiList)
            prevPartialMatchedPhrases = Util.addUnique(candidatePhrases, prevPartialMatchedPhrases)
        return cuiList  
    
    def exactTokenMatchCondition(self, phrase, candidatePhrases, terminology, cuiList):
        for candidatePhrase in candidatePhrases:
            if not Ling.exactTokenMatch(candidatePhrase, phrase):
                continue

            cui = terminology.getNameToCuiListMap().get(candidatePhrase)[0]
            cuiList = Util.setList(cuiList, cui)
                   
        return cuiList
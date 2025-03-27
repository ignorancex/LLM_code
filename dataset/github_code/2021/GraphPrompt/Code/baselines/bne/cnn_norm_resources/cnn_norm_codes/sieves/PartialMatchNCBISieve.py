
import util.Concept
import util.Ling
import Terminology
import Util


class PartialMatchNCBISieve(): 
    def __init__(self):
        pass
    
    def apply(self, concept):
        name = concept.getName()
        nameTokens = name.split(" ")
        return self.partialMatch(name, nameTokens)
    
        
    def partialMatch(self, phrase, phraseTokens):
        partialMatchedPhrases = []
        candidateCuiDataMap = self.init()
        
        for phraseToken in phraseTokens:
            if phraseToken in Ling.getStopwordsList():
                continue;
            candidatePhrases = None
            map = -1
            
            if Sieve.getTrainingDataTerminology().getTokenToNameListMap().__contains__(phraseToken):
                candidatePhrases = Sieve.getTrainingDataTerminology().getTokenToNameListMap().get(phraseToken)
                map = 2
            
            elif  Sieve.getStandardTerminology().getTokenToNameListMap().__contains__(phraseToken):
                candidatePhrases = Sieve.getStandardTerminology().getTokenToNameListMap().get(phraseToken)
                map = 3
            
            if candidatePhrases == None:
                continue
                        
            candidatePhrases.removeAll(partialMatchedPhrases)
            
            candidateCuiDataMap = ncbiPartialMatch(phrase, candidatePhrases, partialMatchedPhrases,Sieve.getTrainingDataTerminology() if map == 2 else Sieve.getStandardTerminology(), candidateCuiDataMap)
            
        return getCui(candidateCuiDataMap.get(1), candidateCuiDataMap.get(2)) if not candidateCuiDataMap[0].isEmpty() else "";
      
    
    def init(self):
        candidateCuiDataMap = {}
        candidateCuiDataMap[1] = {}
        candidateCuiDataMap[2] = {}
        return candidateCuiDataMap
    
    def ncbiPartialMatch(phrase, candidatePhrases, partialMatchedPhrases, terminology, cuiCandidateDataMap):
        cuiCandidateMatchingTokensCountMap = cuiCandidateDataMap[1]
        cuiCandidateLengthMap = cuiCandidateDataMap[2]
        
        for  candidatePhrase in candidatePhrases:
            partialMatchedPhrases = Util.setList(partialMatchedPhrases, candidatePhrase)

            count = Ling.getMatchingTokensCount(phrase, candidatePhrase)

            length = candidatePhrase.split(" ").length
            cui = terminology.getNameToCuiListMap().get(candidatePhrase)[0]

            if cuiCandidateMatchingTokensCountMap.__contains__(cui):
                oldCount = cuiCandidateMatchingTokensCountMap[cui]
                if oldCount < count:
                    cuiCandidateMatchingTokensCountMap[cui] = count
                    cuiCandidateLengthMap[cui] =  length
                continue

            cuiCandidateMatchingTokensCountMap[cui] =  count
            cuiCandidateLengthMap[cui] =  length                 
        
        cuiCandidateDataMap[1] = cuiCandidateMatchingTokensCountMap

        cuiCandidateDataMap[2] = cuiCandidateLengthMap
        return cuiCandidateDataMap  
    
    def getCui(cuiCandidateMatchedTokensCountMap,  cuiCandidateLengthMap):
        cui = ""
        maxMatchedTokensCount = -1
        matchedTokensCountCuiListMap = {}
        for candidateCui in cuiCandidateMatchedTokensCountMap.keys():
            matchedTokensCount = cuiCandidateMatchedTokensCountMap[candidateCui]
            if matchedTokensCount >= maxMatchedTokensCount:
                maxMatchedTokensCount = matchedTokensCount              
                cuiList = matchedTokensCountCuiListMap[matchedTokensCount]
                if cuiList == None:
                    cuiList = []
                    matchedTokensCountCuiListMap[matchedTokensCount] = cuiList
                   
                cuiList = Util.setList(cuiList, candidateCui)
    
        candidateCuiList = matchedTokensCountCuiListMap[maxMatchedTokensCount]

        if len(candidateCuiList) == 1:
            return candidateCuiList[0]
        else:
            minCandidateLength = 1000
            for candidateCui in candidateCuiList:
                length = cuiCandidateLengthMap[candidateCui]
                if length < minCandidateLength:
                    minCandidateLength = length
                    cui = candidateCui
        return cui
import os 
from ling import *
import re
from re import search

class Utils():
    def __init__(self) -> None:
        self.ling = Ling()
        
    def split_string_into_prepostional_phrases(self, string_):
        prepositions_list = ["in", "with", "on", "of"]
        all_phrases = []
        word_list_string = string_.split(" ")
        total_words = len(word_list_string)
        for word_idx, word_ in enumerate(word_list_string):
            if str(word_list_string[word_idx]) in prepositions_list and word_idx !=0 and word_idx!=total_words-1:
                current_pp_phrase = [word_list_string[word_idx-1], word_list_string[word_idx], word_list_string[word_idx+1]]
                if word_idx !=total_words-1 and word_list_string[word_idx+1] == "the":
                    current_pp_phrase = [word_list_string[word_idx-1], word_list_string[word_idx], word_list_string[word_idx+1] + " " + word_list_string[word_idx+2]]
                all_phrases.append(current_pp_phrase)  
        return all_phrases 

    
    def make_string_from_str_list(self, str_list):
        # makes string from list of words with space as delimited between words
        result_string = ""
        for word_ in str_list:
            result_string = result_string + word_ + " "
        result_string.strip()
        return result_string

    def swap_positions(self, list_, pos1, pos2):
        list_[pos1], list_[pos2] = list_[pos2], list_[pos1]
        return list_

    def get_expanded_word(self, item_dict, query):
        try:
            long_form_word  = item_dict[query]
        except:
            long_form_word = None
        return long_form_word

    def merge_dict(self, dict1, dict2):
        res = {**dict1, **dict2}
        return res

    def concat_abbrevations_files(self, file_path_list = [os.path.join("../text_resources/semeval-wiki-abbreviations.txt"), os.path.join("../text_resources/ncbi-wiki-abbreviations.txt")]):
        all_abbrevations_dict  = {}
        for file_path in file_path_list:
            file_handle  = open(file_path,"r")
            for line in file_handle:
                line  = line.replace("\n","")
                line_split  = line.split("||")
                all_abbrevations_dict[line_split[0]] = line_split[1]
        return all_abbrevations_dict

    def get_numbers_mappings(self):
        number_mapping = {}
        # initialize numbers mapping
        for i in range(1,11):
            number_mapping[str(i)] = []
        f = open(os.path.join("../text_resources/number.txt"),"r")
        for line_ in f:
            line_ = str(line_).replace("\n","")
            line_split = line_.split("||")
            number_mapping[line_split[0]].append(line_split[1])
        
        number_mapping_list = []
        for item_ in number_mapping:
            list_ = [item_]
            list_.extend(number_mapping[item_])
            number_mapping_list.append(list_)

        all_mapped_items_list  = []
        for list_ in number_mapping_list:
            all_mapped_items_list.extend(list_)
        return number_mapping, number_mapping_list, all_mapped_items_list
    
    def get_number_group(self, word_, number_mapping_list):
        for idx, group in enumerate(number_mapping_list):
            if word_ in group:
                return idx

    def get_hyphenated_string(self, words_query):
        hyphenated_strings_list = []
        for i in range(1, len(words_query)):
            hyphenated_string = ""
            for j in range(0,len(words_query)):
                if j==i:
                    hyphenated_string+= "-" + words_query[j]
                else:
                    if hyphenated_string == "":
                        hyphenated_string = words_query[j]
                    else:
                        hyphenated_string = hyphenated_string + " " + words_query[j]
            hyphenated_strings_list.append(hyphenated_string)
        return hyphenated_strings_list
    
    def get_dehyphenated_string(self, words_query):
        dehyphenated_strings_list = []
        for i in range(1, len(words_query)):
            dehyphenated_string = ""
            for j in range(0,len(words_query)):
                if j==i:
                    dehyphenated_string+= " " + words_query[j]
                else:
                    if dehyphenated_string == "":
                        dehyphenated_string = words_query[j]
                    else:
                        dehyphenated_string = dehyphenated_string + " " + words_query[j]
            dehyphenated_strings_list.append(dehyphenated_string)
        return dehyphenated_strings_list

    def get_suffixation_combinations(self, stringTokens):
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
    
    def getTokenIndex (self, tokens, token):
        i = 0
        while i < len(tokens):
            if tokens[i] == token:
                return i
            i =  i + 1
        return -1
    
    def getModifier(self, stringTokens, modifiers):
        for modifier in modifiers:
            index = self.getTokenIndex(stringTokens, modifier)
            if index != -1:
                return stringTokens[index]
        return ""
    
    def setList(self, list_, value):
        if not (value in list_) and value!="":
            list_.append(value)
        return list_

    
    def addUnique(self, list_, newList):
        for value in newList:
            list_ = self.setList(list_, value)
        return list_

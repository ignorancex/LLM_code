import os
import sys
sys.path.insert(0, "/home/megh/projects/entity-norm/cnn-norm-obo/")
from code.dataset import *
from utils import Utils
import numpy as np
import torch
from ling import * 

class Sieves():
    def __init__(self, dataset_path = os.path.join("../../data/datasets/cl.obo"), util = None) -> None:
        self.name_array, self.query_id_array, self.mention2id, self.edge_index  = load_data(filename=dataset_path)
        self.name2id, self.query2id  = self.process_data_for_seive()
        self.normalized_queries = {}
        self.current_unnormalized_queries = self.query2id.copy()
        self.unnormalized_queries_final  = {}
        self.util = util
        self.ling = Ling()

    def process_data_for_seive(self):
        name2id = {}
        for name_ in self.name_array:
            id_number =  self.mention2id[str(name_)]
            name2id[str(name_)] = id_number
        query2id  = {}
        for item in self.query_id_array:
            query2id[str(item[0])] =  int(item[1])
        return name2id, query2id

    def exact_match_util(self, query_all_forms_list, query_original_form):
        for name_ in self.name2id.keys():
            name_ = str(name_)
            if name_ in query_all_forms_list:
                self.normalized_queries[query_original_form] = self.query2id[query_original_form]
                del self.current_unnormalized_queries[query_original_form]

    def sieve_exact_match(self):
        self.current_unnormalized_queries_for_iteration = self.current_unnormalized_queries.copy()
        for query_ in self.current_unnormalized_queries_for_iteration.keys():
            self.exact_match_util([query_],query_)
        

    def abbrevation_expansion_sieve(self):
        self.all_abbrevations_dict  = self.util.concat_abbrevations_files()
        # this dict will be appended to the original dict after all the expanded forms are found
        self.current_unnormalized_queries_for_iteration = self.current_unnormalized_queries.copy()
        for query_ in self.current_unnormalized_queries_for_iteration.keys():
            expanded_query_dict  = {}
            # copy string into another string
            query_expanded_copy = (query_ + ".")[:-1]
            # splitting the words in a query 
            words_query = str(query_).split(" ")
            for word_ in words_query:
                expanded_word = self.util.get_expanded_word(self.all_abbrevations_dict,str(word_))
                if expanded_word is None:
                    continue
                query_expanded_copy = query_expanded_copy.replace(str(word_),str(expanded_word))
                expanded_query_dict[query_expanded_copy] = self.current_unnormalized_queries[query_]
            self.exact_match_util(expanded_query_dict.keys(), query_)



    def subject_object_sieve(self):
        prepositions_list = ["in", "with", "on", "of"]
        self.current_unnormalized_queries_for_iteration = self.current_unnormalized_queries.copy()

        for query_ in self.current_unnormalized_queries_for_iteration.keys():
            # replacing one preposition with another and forming all possible permutations
            form_1_query_dict = {}
            words_query =  query_.split(" ")
            query_copy = (query_ + " ")[:-1]
            for word in words_query:
                if str(word) not in prepositions_list:
                    continue
                elif str(word) in prepositions_list:
                    for preposition in prepositions_list:
                        if str(preposition) == str(word):
                            continue
                        elif str(preposition) != str(word):
                            query_copy.replace(str(word),str(preposition))
                            form_1_query_dict[query_copy] = self.current_unnormalized_queries[query_]
            self.exact_match_util(form_1_query_dict.keys(),query_copy)
            
            # dropping preposition and swapping the substrings surrounding it
            form_2_query_dict = {}
            word_query = query_.split(" ") 
            
            for word_idx, word in enumerate(word_query):
                word_query_copy = word_query.copy()
                if str(word) in prepositions_list and word_idx!=0:
                    self.util.swap_positions(word_query_copy, word_idx-1, word_idx+1)
                    word_query_copy.pop(word_idx)
                    string_new  = self.util.make_string_from_str_list(word_query_copy)
                    form_2_query_dict[string_new] = self.current_unnormalized_queries[query_]

            self.exact_match_util(form_2_query_dict.keys(),query_copy)

            # form 3 modifications
            form_3_query_dict = {}
            original_prepositional_phrases = self.util.split_string_into_prepostional_phrases(query_)

            # form 4 modifications
            # write code here
            form_4_query_dict = {}
          

    def numbers_replacement_sieve(self):
        number_mapping, number_mapping_list, all_mapped_items_list = self.util.get_numbers_mappings()
        self.current_unnormalized_queries_for_iteration = self.current_unnormalized_queries.copy()
        for query_ in self.current_unnormalized_queries_for_iteration.keys():
            word_query = query_.split(" ")
            query_copy = (query_ + ".")[:-1]
            form_replacement_numbers  = {}
            for word_ in word_query:
                if word_ in all_mapped_items_list:
                    number_group_idx = self.util.get_number_group(word_, number_mapping_list) # always returns something since that condition is checked before
                    for item_ in number_mapping_list[number_group_idx]:
                        if str(item_) == word_:
                            continue
                    new_string  = query_copy.replace(word_, str(item_))
                    form_replacement_numbers[new_string] = self.current_unnormalized_queries[query_]
            self.exact_match_util(form_replacement_numbers.keys(),query_)
    
    def hyphenation_sieve(self):
        self.current_unnormalized_queries_for_iteration = self.current_unnormalized_queries.copy()
        for query_ in self.current_unnormalized_queries_for_iteration:
            # hyphenation - add hyphens if the query does not contain any hyphens
            words_query = query_.split(" ")
            hyphenated_strings_list = self.util.get_hyphenated_string(words_query)
            dehyphenated_strings_list =  self.util.get_dehyphenated_string(words_query)
            self.exact_match_util(hyphenated_strings_list, query_)     
            self.exact_match_util(dehyphenated_strings_list, query_)

    def affix(self):
        self.current_unnormalized_queries_for_iteration = self.current_unnormalized_queries.copy()
        for query_ in self.current_unnormalized_queries_for_iteration.keys():
            word_query = query_.split(" ")
            new_phrases = self.util.suffixation(word_query, query_)
            new_phrases.append(self.util.prefixation(word_query, query_))
            new_phrases.append(self.util.affixation(word_query, query_))     
            self.exact_match_util(new_phrases,query_)

    def disorder_syn_replacement(self):
        self.current_unnormalized_queries_for_iteration =  self.current_unnormalized_queries.copy()
        transformedNames = []
        for nameForTransformation in self.current_unnormalized_queries_for_iteration:
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
            self.exact_match_util(transformedNames, nameForTransformation)

    def stemming(self):
        pass
        
    def composite_name(self):
        pass
        
    def partial_match_sieve(self):
        pass

    def analysis_dataset(self):
        prepositions_list  = ["in", "with", "on", "of"]
        preposition_count_list = []
        for query_ in self.mention2id.keys():
            preposition_count = 0
            word_query = query_.split(" ")
            for word_ in word_query:
                if str(word_) in prepositions_list:
                    preposition_count = preposition_count+1
            preposition_count_list.append(preposition_count)
        return
    
    
    
    def run_sieves(self):
        
        self.sieve_exact_match()
        self.abbrevation_expansion_sieve()
        self.subject_object_sieve()
        self.numbers_replacement_sieve()
        self.hyphenation_sieve()
        self.affix()
        self.disorder_syn_replacement()
        self.stemming()
        self.composite_name()
        self.partial_match()




if __name__ == '__main__':
    util = Utils()
    s_classifier = Sieves(util = util)
    s_classifier.run_sieves()

    
    
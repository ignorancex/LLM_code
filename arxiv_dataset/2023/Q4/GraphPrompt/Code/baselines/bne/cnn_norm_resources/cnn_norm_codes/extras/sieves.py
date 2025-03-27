import os
import sys
sys.path.insert(0,"/home/megh/projects/entity-norm/syn/")
from code.dataset import *
import numpy as np
import torch

class Sieves():

    def __init__(self, dataset_path = os.path.join("../../data/datasets/cl.obo")) -> None:
        self.name_array, self.query_id_array, self.mention2id, self.edge_index  = load_data(filename=dataset_path)
        self.name2id, self.query2id  = self.process_data_for_seive()
        self.normalized_queries = {}
        self.current_unnormalized_queries = self.query2id.copy()
        self.unnormalized_queries_final  = {}
        self.all_forms_queries_dict = {}
        for key_ in self.query2id.keys():
            self.all_forms_queries_dict[key_] = []


    def process_data_for_seive(self):
        name2id = {}
        for name_ in self.name_array:
            id_number =  self.mention2id[str(name_)]
            name2id[str(name_)] = id_number
        query2id  = {}
        for item in self.query_id_array:
            query2id[str(item[0])] =  int(item[1])
        return name2id, query2id


    def exact_match_sieve(self):
        for query_ in self.current_unnormalized_queries.keys():
            for name_ in self.name_array:
                query_ = str(query_)
                name_ = str(name_)
                if query_ == name_:
                    print("Found exact match")
                    self.normalized_queries[query_] = self.current_unnormalized_queries[query_].copy()
            # deleting the entries from the unnormalized queries
            for key_ in self.normalized_queries:
                del self.current_unnormalized_queries[key_]
            return 

    def concat_abbrevations_files(self, file_path_list = [os.path.join("../text_resources/semeval-wiki-abbreviations.txt"), os.path.join("../text_resources/ncbi-wiki-abbreviations.txt")]):
        all_abbrevations_dict  = {}
        for file_path in file_path_list:
            file_handle  = open(file_path,"r")
            for line in file_handle:
                line  = line.replace("\n","")
                line_split  = line.split("||")
                all_abbrevations_dict[line_split[0]] = line_split[1]
        return all_abbrevations_dict

    def get_expanded_word(self, item_dict, query):
        try:
            long_form_word  = item_dict[query]
        except:
            long_form_word = None
        return long_form_word

    def merge_dict(self, dict1, dict2):
        res = {**dict1, **dict2}
        return res

    def abbrevation_expansion_sieve(self):
        self.all_abbrevations_dict  = self.concat_abbrevations_files()

        # this dict will be appended to the original dict after all the expanded forms are found
        expanded_query_dict  = {}
        for query_ in self.current_unnormalized_queries.keys():
            # copy string into another string
            query_expanded_copy = (query_ + ".")[:-1]
            # splitting the words in a query 
            words_query = str(query_).split(" ")
            for word_ in words_query:
                expanded_word = self.get_expanded_word(self.all_abbrevations_dict,str(word_))
                if expanded_word is None:
                    continue
                query_expanded_copy = query_expanded_copy.replace(str(word_),str(expanded_word))

                expanded_query_dict[query_expanded_copy] = self.current_unnormalized_queries[query_]
                self.all_forms_queries_dict[str(query_)].append(query_expanded_copy)

        self.current_unnormalized_queries =  self.merge_dict(self.current_unnormalized_queries, expanded_query_dict)
        self.exact_match_sieve()
        return

    def swap_positions(self, list_, pos1, pos2):
        list_[pos1], list_[pos2] = list_[pos2], list_[pos1]
        return list_

    def make_string_from_str_list(self, str_list):
        # makes string from list of words with space as delimited between words
        result_string = ""
        for word_ in str_list:
            result_string = result_string + word_ + " "
        result_string.strip()
        return result_string

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


    def subject_object_sieve(self):
        prepositions_list = ["in", "with", "on", "of"]
        form_1_query_dict = {}

        for query_ in self.current_unnormalized_queries.keys():
            # replacing one preposition with another and forming all possible permutations
            words_query =  query_.split(" ")
            query_copy = (query_ + " ")[:-1]
            for word in words_query:
                replaced_string_list = []
                if str(word) not in prepositions_list:
                    continue
                elif str(word) in prepositions_list:
                    for preposition in prepositions_list:
                        if str(preposition) == str(word):
                            continue
                        elif str(preposition) != str(word):
                            query_copy.replace(str(word),str(preposition))
                            replaced_string_list.append(query_copy)
                            form_1_query_dict[query_copy] = self.current_unnormalized_queries[query_]
                            self.all_forms_queries_dict[query_].append(query_copy)
            
            # dropping preposition and swapping the substrings surrounding it
            form_2_query_dict = {}
            word_query = query_.split(" ") 
            word_query_copy = word_query.copy()
            print(word_query_copy)
            for word_idx, word in enumerate(word_query):
                if str(word) in prepositions_list and word_idx!=0:
                    self.swap_positions(word_query_copy, word_idx-1, word_idx+1)
                    word_query_copy.pop(word_idx)
                    string_new  = self.make_string_from_str_list(word_query_copy)
                    form_2_query_dict[string_new] = self.current_unnormalized_queries[query_]
                    self.all_forms_queries_dict[query_].append(string_new)
            
            # bringing last token front and dropping the preposition

            original_prepositional_phrases = self.split_string_into_prepostional_phrases(query_)
            new_phrases  = []

            for idx, phrase in enumerate(original_prepositional_phrases):
                new_phrase  = [phrase[2],phrase[0]]
                new_phrases.append(new_phrase)
    

            print(new_phrases)





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
        self.exact_match_sieve()
        self.abbrevation_expansion_sieve()
        self.subject_object_sieve()

if __name__ == '__main__':
    s_classifier = Sieves()
    s_classifier.run_sieves()
    
    
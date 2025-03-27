"""
this file contains all the models
"""
import Levenshtein
import numpy as np
from tqdm import tqdm
from evaluator import Evaluator
#edit distance model, calulate the similarity accroding edit distance
class EditDistance_Classifier():
    def __init__(self,concepts_list) :
        self.concepts_list = concepts_list
    
    def softmax(self,array):
        exp_res = np.exp(array)
        sum_exp = np.sum(exp_res)
        return exp_res/sum_exp

    #use mean_distance/distance as the similarity score
    def forward(self,mentions):
        """
        return: the score arrays of all concepts over all samples
        """
        score_matrix=[]
        for j,mention in tqdm(enumerate(mentions)):
            distance_array=np.ones(len(self.concepts_list))
            for i,concept in enumerate(self.concepts_list):
                distance_array[i] = Levenshtein.distance(mention,concept) + 1#in case that edit distance equals to zero
            score_array =  self.softmax(np.divide(distance_array.mean(),distance_array))
            score_matrix.append(score_array)
        score_matrix = np.stack(score_matrix,axis=0)#shape:n_samples * n_concepts
        #print(score_matrix.shape)
        
        return score_matrix
    
    def eval(self,mentions,concepts):
        """
        inputss: mentions and concepts are both list of string 
        """
        score_matrix = self.forward(mentions)
        true_labels = np.array([self.concepts_list.index(concept) for concept in concepts])
        evaluator = Evaluator()
        accu1 = evaluator.accu(score_matrix,true_labels,top_k=1)
        accu5 = evaluator.accu(score_matrix,true_labels,top_k=5)
        return accu1,accu5
        









        
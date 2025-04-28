"""
This module produces fused protein representation combinations from existing protein representations. 
The module structure is the following:

- The module implements a common ``produce_fused_representations`` method. 
The method takes a list of protein representations and fuse them.
The fused representations are both saved and returned by the function.
"""

import pandas as pd
from itertools import *
import os
from sklearn.utils import shuffle


def produce_fused_representations(
    list_of_protein_representation_dataframes,
    minimum_number_of_combinations,
    representation_name_list,
):

    """
    This function takes a number protein representations and fuse them. Multi combinations of these representations can be created.
    A list for dataframes as input. Each dataframe has two columns 'Entry' and 'Vector'.
    Entry includes UniProt protein ids and Vector includes the protein representations vectors.
    Function produces combinations according to minimum_number_of_combinations
    parameter. For example if 3 representations are supplied and minimum_number_of_combinations = 2
    then the function will produce double and triple combinations of the protein representation vector.
        Such as Vec1_Vec2, Vec1_Vec3, Vec2_Vec3 and Vec1_Vec2_Vec3
    Parameters
    ----------
    list_of_protein_representation_dataframes: list
            List of protein representation dataframes.
    minimum_number_of_combinations: int
            Minimum number of combination vector that will fused.
    representation_name_list: list
            Representation names list
    Returns
    -------
    fused_dataframes : list
            The list of fused dataframes. Each dataframe has  'Entry' and  multicolumn representation Vector
            The saved dataframes has a different structure than the returned ones. The saved dataframes has a coloumn named as 'Entry',
            but the following columns are a varying number based on representation vector length.
    """
    path=os.getcwd()
    
    if 'case_study_results' in os.listdir(os.path.join(path,"case_study")):
        paths=os.path.join(path,"case_study/case_study_results")     
    else:
        paths=os.path.join(path,"case_study/case_study_results")    
        os.makedirs(os.path.join(path,"case_study/case_study_results"),exist_ok=True)
         
        
    if minimum_number_of_combinations == len(list_of_protein_representation_dataframes):
        fused_dataframes = list_of_protein_representation_dataframes[0]
        for dataset in list_of_protein_representation_dataframes[1:]:
            fused_dataframes = fused_dataframes.merge(dataset, on="Entry")

        name = '_'.join(representation_name_list)

        #import pdb
        #pdb.set_trace()
        
        #fused_dataframes.to_csv(os.path.join(paths, name +'.csv'),index=False)
        return fused_dataframes
    else:
     
        items = [
            item for item in range(0, len(list_of_protein_representation_dataframes))
        ]

        iterable = chain.from_iterable(
            combinations(items, r)
            for r in range(
                minimum_number_of_combinations,
                len(list_of_protein_representation_dataframes) + 1,
            )
        )
        fused_dataframes = []
        lst = []
        for setting in product(iterable):
            lst.append(setting)

        for dataset_ids in lst:
            fused_list = list_of_protein_representation_dataframes[dataset_ids[0][0]]
            fused_name = representation_name_list[dataset_ids[0][0]]
            
            for dataset_id in dataset_ids[0][1:]:
                
                fused_list = fused_list.merge(
                    list_of_protein_representation_dataframes[dataset_id], on="Entry"
                )
                fused_name = fused_name + "_" + representation_name_list[dataset_id]

            # print(fused_list)
            # name = '_'.join(representation_name_list)
            breakpoint()
            print(fused_name)
            
            
            #if "results" not in os.listdir(path):
            #    os.makedirs(path, exist_ok=True)
            
            #fused_list.to_csv( os.path.join(os.getcwd(),"case_study/case_study_results", fused_name + '.csv'))
            # df.to_csv('fused/' + name + ".csv")
            fused_dataframes.append(fused_list)
        return fused_dataframes

import os
import pickle
import datetime
import tqdm
import pandas as pd
from pathlib import Path
path = os.path.join(os.getcwd(), "case_study") 

def integrate_go_lables_and_representations_for_binary(
    label_dataframe, representation_dataframe, dataset_names
):
    """
        This function takes two dataframes and a dataframe name as input. First dataframe has two coloumns 'Label' and  'Entry'.
    'Label' column includes GO Terms and 'Entry' column includes UniProt IDs. Second dataframe has a column named as
    'Entry' in the first column, but number of the the following columns are varying based on representation vector length.
    The function integrates these dateframes based on 'Entry' column

        Parameters
        ----------
        label_dataframe: Pandas Dataframe
                Includes GO Terms and UniProt IDs of annotated proteins.
        representation_dataframe: Pandas Dataframe
                UniProt IDs of annotated proteins and representation vectors in multicolumn format.
        dataset_names: String
                The name of the output dataframe which is used for saving the dataframe to the results directory.

        Returns
        -------
        integrated_dataframe : Pandas Dataframe
                Integrated label dataframe. The dataframe has multiple number of  columns 'Label','Entry' and 'Vector'.
                'Label' column includes GO Terms and 'Entry' column includes UniProt ID and the following columns includes features of the protein represention vector."""
    
    integrated_dataframe = pd.DataFrame(columns=["Entry", "Vector"])    
    integrated_dataframe_list = []
    representation_cols = representation_dataframe.iloc[:, 1 : (len(representation_dataframe.columns))]
    representation_dataset = pd.DataFrame(columns=["Entry", "Vector"])
    for index, row in tqdm.tqdm(representation_cols.iterrows(), total=len(representation_cols)):
        list_of_floats = [float(item) for item in list(row)]
        representation_dataset.loc[index] = [representation_dataframe.iloc[index]["Entry"]] + [
            list_of_floats
        ]
   
    integrated_dataframe = label_dataframe.merge(
            representation_dataset, how="inner", on="Entry"
        )
    integrated_dataframe.drop(
            integrated_dataframe.filter(regex="Unname"), axis=1, inplace=True
        )
    
    #breakpoint()
    path_binary_data = os.path.join(os.getcwd(), "case_study/case_study_results",dataset_names+"_binary_data.pickle") 
    with open(path_binary_data,
            "wb",
        ) as handle:
            pickle.dump(integrated_dataframe, handle)
            
    return integrated_dataframe

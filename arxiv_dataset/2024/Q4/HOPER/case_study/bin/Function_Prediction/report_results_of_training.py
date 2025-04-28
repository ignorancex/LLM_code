""" -The module implements `report_results_of_training`` method. 
The method takes representation_names,result list and mean result list as input and 
writes down results to files in a proper format ."""

import pandas as pd
import os


def report_results_of_training(
    representation_name,
    result_list,
    mean_result_list,
    classifier_name,
    file_name,
    index,
    classifier_type,
):
    path = os.path.join(os.getcwd(),"case_study/case_study_results")
    if "test" not in os.listdir(path):
        os.makedirs(path + classifier_type, exist_ok=True)
    # inex-1=len(class_len)
    result_len = 27
    if index == 1:
        result_df = pd.DataFrame(
            columns=[
                "representation_name",
                "classifier_name",
                "accuracy",
                "std_accuracy",
                "f1_micro",
                "std_f1_micro",
                "f1_macro",
                "std_f1_macro",
                "f_max",
                "std_f_max",
                "f1_weighted",
                "std_f1_weighted",
                "precision_micro",
                "std_precision_micro",
                "precision_macro",
                "std_precision_macro",
                "precision_weighted",
                "std_precision_weighted",
                "recall_micro",
                "std_recall_micro",
                "recall_macro",
                "std_recall_macro",
                "recall_weighted",
                "std_recall_weighted",
                "hamming distance",
                "std_hamming distance",
                "auc",
                "std_auc",
                "matthews correlation coefficient",
            ]
        )
        result_df.loc[index - 1] = result_list[0]
        result_df.reset_index(drop=True, inplace=True)
        mean_result_dataframe = pd.DataFrame(
            columns=[
                "representation_name",
                "classifier_name",
                "accuracy",
                "std_accuracy",
                "f1_micro",
                "std_f1_micro",
                "f1_macro",
                "std_f1_macro",
                "f_max",
                "std_f_max",
                "f1_weighted",
                "std_f1_weighted",
                "precision_micro",
                "std_precision_micro",
                "precision_macro",
                "std_precision_macro",
                "precision_weighted",
                "std_precision_weighted",
                "recall_micro",
                "std_recall_micro",
                "recall_macro",
                "std_recall_macro",
                "recall_weighted",
                "std_recall_weighted",
                "hamming distance",
                "std_hamming distance",
                "auc",
                "std_auc",
                "matthews correlation coefficient",
            ]
        )

        mean_result_dataframe.loc[index - 1] = mean_result_list[0]
        mean_result_dataframe.reset_index(drop=True, inplace=True)
        result_df.to_csv(os.path.join(path,classifier_type,representation_name+ "_"+ file_name+ "_"+ classifier_type+ ".tsv"),sep="\t",index=False,)

        mean_result_dataframe.to_csv(os.path.join(path,classifier_type,representation_name+"_"+file_name+ "_"+ classifier_type+ "_means.tsv"),sep="\t",index=False,)

    # rep_name="_".join(representation_name)

    elif index > 1:

        if index > 2:
            result_df = pd.read_csv(os.path.join(path,classifier_type,representation_name+"-"+file_name+"_"+classifier_type+".tsv"),sep="\t",)
            result_df.loc[index - 1] = result_list[0]
            result_df.reset_index(drop=True, inplace=True)
            result_df.to_csv(os.path.join(path,classifier_type,representation_name+"_"+file_name+"_"+classifier_type+".tsv"),sep="\t",index=False,)


            mean_result_dataframe = pd.read_csv(os.path.join(path,classifier_type,representation_name+file_name+"_"+classifier_type+"_means.tsv"),sep="\t",)

            mean_result_dataframe.loc[index - 1] = mean_result_list
            mean_result_dataframe.reset_index(drop=True, inplace=True)
            mean_result_dataframe.to_csv(os.path.join(path,classifier_type,representation_name+file_name+"_"+classifier_type+"_means.tsv"),sep="\t",index=False,)
              
        else:
            result_df = pd.read_csv(os.path.join(path,classifier_type,representation_name+ "_"+ file_name+ "_"+ classifier_type+ ".tsv"),sep="\t")

            result_df.loc[index - 1] = result_list[0]
            result_df.reset_index(drop=True, inplace=True)

            result_df.to_csv(os.path.join(path,classifier_type,representation_name+ "_"+ file_name+ "_"+ classifier_type+ ".tsv"),sep="\t",index=False,)

            mean_result_dataframe = pd.read_csv(os.path.join(path,classifier_type,representation_name+ "_"+ file_name+ "_"+ classifier_type+ "_means.tsv"),sep="\t",)

            mean_result_dataframe.loc[index - 1] = mean_result_list[0]
            mean_result_dataframe.reset_index(drop=True, inplace=True)

            mean_result_dataframe.to_csv(os.path.join(path,classifier_type,representation_name+ file_name+ "_"+ classifier_type+ "_means.tsv"),index=False,)
                
    pass

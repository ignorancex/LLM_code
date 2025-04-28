import pandas as pd
import glob
import os
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#import pixiedust
pd.set_option('display.max_colwidth', -1)
pd.options.display.max_rows = 4000

result_path = ''

#Classic Representations: Yellow
#Learned Small Scale Models: Blue
#Learned Large Scale Models: Red
group_color_dict = {'TFIDF-UNIPROT-256':'orange','TFIDF-UNIPROT-512':'orange','TFIDF-UNIPROT-1024':'orange','TFIDF-UNIPROT-PUBMED-256':'green','TFIDF-UNIPROT-PUBMED-512':'red', 'TFIDF-UNIPROT-PUBMED-1024':'red',\
                    'UNIPROT-BERT-AVG':'red', 'UNIPROT-BERT-POOL':'red','UNIPROT-BIOSENTVEC':'green', 'UNIPROT-BIOWORDVEC':'red', 'UNIPROT-PUBMED-BERT-AVG':'red', 'UNIPROT-PUBMED-BERT-POOL':'green',\
                    'UNIPROT-PUBMED-BIOSENTVEC':'green','UNIPROT-PUBMED-BIOWORDVEC':'red'}

#create_index_from_model_name(index_names): Creates an index list from model names.

def create_index_from_model_name(index_names):
    index_list = []
    for index_name in index_names:
        new_name = index_name.split("_")[1:len(index_names)]
        new_name = '_'.join(new_name)
        index_list.append(new_name)
    return index_list

#create_pred_table(measure): Reads prediction results, orders them alphabetically, and creates a prediction table.
def create_pred_table(measure):
       
    go_pred_table = pd.DataFrame()
    for filename in sorted(glob.glob(os.path.join(result_path, '*_5cv_mean_*'))):
            col_name = filename.split("Ontology_based_function_prediction_5cv_mean_")[-1].split(".")[0]
            #print(col_name)
            tmp_column = pd.read_csv(filename,sep="\t")
            tmp_column.sort_values(tmp_column.columns[0])

            go_pred_table[col_name] = tmp_column[measure]
            index = create_index_from_model_name(list(tmp_column.iloc[:, 0]))
    #print(go_pred_table)
    go_pred_table["index_col"] = index
    go_pred_table.set_index('index_col', inplace=True)
    go_pred_table.sort_index(inplace=True)
    #print(go_pred_table)
    return go_pred_table

# get_go_pred_table_for_aspect(aspect, go_pred_table): Slices the prediction table by aspect and orders subgroups.
def get_go_pred_table_for_aspect(aspect,go_pred_table):
    
    if aspect == "BP":
        go_pred_tableBP = go_pred_table[0:9]
        
        new_index =  ["BP_High_Shallow", "BP_High_Normal", "BP_High_Specific",\
                      "BP_Middle_Shallow","BP_Middle_Normal","BP_Middle_Specific",\
                      "BP_Low_Shallow","BP_Low_Normal","BP_Low_Specific"]
        go_pred_tableBP = go_pred_tableBP.reindex(new_index)
        #print(go_pred_table)
        return go_pred_tableBP
    if aspect == "CC":
        go_pred_tableCC = go_pred_table[9:17]
        new_index =  ["CC_High_Shallow", "CC_High_Normal",\
                      "CC_Middle_Shallow","CC_Middle_Normal","CC_Middle_Specific",\
                      "CC_Low_Shallow","CC_Low_Normal","CC_Low_Specific"]
        go_pred_tableCC = go_pred_tableCC.reindex(new_index) 
       
        return go_pred_tableCC
    if aspect == "MF":
        go_pred_tableMF = go_pred_table[17:25]
        new_index =  ["MF_High_Shallow", "MF_High_Normal",\
                      "MF_Middle_Shallow","MF_Middle_Normal","MF_Middle_Specific",\
                      "MF_Low_Shallow","MF_Low_Normal","MF_Low_Specific"]
        go_pred_tableMF = go_pred_tableMF.reindex(new_index) 
       
        return go_pred_tableMF


#prepare_figure_data_for_aspect(aspect): Calculates mean measures for different aspects and returns F1 weighted scores.
def prepare_figure_data_for_aspect(aspect):
    go_pred_tableF1 = create_pred_table("F1_Weighted")
    go_pred_tableACC = create_pred_table("Accuracy")
    go_pred_tablePR = create_pred_table("Precision_Weighted")
    go_pred_tableREC = create_pred_table("Recall_Weighted")
    go_pred_tableHAMM = create_pred_table("Hamming_Distance")
    
    go_pred_tableF1_aspect = get_go_pred_table_for_aspect(aspect,go_pred_tableF1)
    go_pred_tableACC_aspect = get_go_pred_table_for_aspect(aspect,go_pred_tableACC)
    go_pred_tablePR_aspect = get_go_pred_table_for_aspect(aspect,go_pred_tablePR)
    go_pred_tableREC_aspect = get_go_pred_table_for_aspect(aspect,go_pred_tableREC)
    go_pred_tableHAMM_aspect = get_go_pred_table_for_aspect(aspect,go_pred_tableHAMM)
    #print(go_pred_tableF1_aspect)
    go_pred_tableF1_aspect_mean = go_pred_tableF1_aspect.mean(axis = 0)
    go_pred_tableACC_aspect_mean = go_pred_tableACC_aspect.mean(axis = 0) 
    go_pred_tablePR_aspect_mean = go_pred_tablePR_aspect.mean(axis = 0) 
    go_pred_tableREC_aspect_mean = go_pred_tableREC_aspect.mean(axis = 0)
    go_pred_tableHAMM_aspect_mean = go_pred_tableHAMM_aspect.mean(axis = 0)
    
    #print(go_pred_tableF1_aspect_mean)


    new_index =  ["Accuracy","F1-Weighted","Precision","Recall", "Hamming"]
    pred_mean_df = pd.DataFrame([go_pred_tableACC_aspect_mean])
    
    pred_mean_df = pred_mean_df.append(go_pred_tableF1_aspect_mean, ignore_index=True)
    pred_mean_df = pred_mean_df.append(go_pred_tablePR_aspect_mean, ignore_index=True)
    pred_mean_df = pred_mean_df.append(go_pred_tableREC_aspect_mean, ignore_index=True)
    pred_mean_df = pred_mean_df.append(go_pred_tableHAMM_aspect_mean, ignore_index=True)
   
    pred_mean_df = pred_mean_df.set_index(pd.Series(new_index))
    pred_mean_df_table = pred_mean_df.transpose()
    #print(pred_mean_df_table)
    path=os.getcwd()
    pred_mean_df_table.to_csv(os.path.join(path,"text_representations/result_visualization/tables/" + aspect + "_pred_mean_table.csv"))

    display_labels = ['INTERPRO2GO','UNIRULE2GO','ENSEMBL-ORTHOLOGY','BLAST','HMMER','K-SEP','APAAC','PFAM','AAC','PROTVEC',\
    'GENE2VEC','LEARNED-VEC','MUT2VEC','TCGA-EMBEDDING','SEQVEC','CPC-PROT','BERT-BFD',\
    'BERT-PFAM','ESMB1','ALBERT','XLNET','UNIREP','T5']
    
    columnsTitles = ['INTERPRO2GO','UNIRULE2GO','ENSEMBL-ORTHOLOGY','BLAST','HMMER','K-SEP','APAAC','PFAM','AAC','PROTVEC',\
    'GENE2VEC','LEARNED-VEC','MUT2VEC','TCGA-EMBEDDING','SEQVEC','CPC-PROT','BERT-BFD',\
    'BERT-PFAM','ESMB1','ALBERT','XLNET','UNIREP','T5']
    
    #pred_mean_df = pred_mean_df.reindex(columns=columnsTitles)
    #go_pred_tableF1_aspect = go_pred_tableF1_aspect.reindex(columns=columnsTitles)
    #go_pred_tablePR_aspect = go_pred_tablePR_aspect.reindex(columns=columnsTitles)
    
    #pred_mean_df.columns = display_labels
    #go_pred_tableF1_aspect.columns = display_labels
    #print(go_pred_tablePR_aspect)
    return pred_mean_df,go_pred_tableF1_aspect,go_pred_tablePR_aspect
	
#set_colors_and_marks_for_representation_groups(ax): Sets colors and marks for representation groups in a plot.	
def set_colors_and_marks_for_representation_groups(ax):
    for label in ax.get_xticklabels():
        label.set_color(group_color_dict[label.get_text()])
        lt = label.get_text()
        if  lt == 'MUT2VEC' or lt =='PFAM' or lt == 'GENE2VEC' or lt == 'BERT-PFAM':
            signed_text = "^" + label.get_text()
            label.set_text(signed_text)
    fontproperties = {'weight' : 'bold'}
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties)

#create_figures(): Creates dataframes for figures and generates the figures.
def create_figures():
    path=os.getcwd()
    #Create dataframes for figures
    pred_mean_df_BP, go_pred_tableF1_BP,go_pred_tablePR_Precision_BP = prepare_figure_data_for_aspect("BP")
    pred_mean_df_CC, go_pred_tableF1_CC,go_pred_tablePR_Precision_CC = prepare_figure_data_for_aspect("CC")
    pred_mean_df_MF, go_pred_tableF1_MF, go_pred_tablePR_Precision_MF = prepare_figure_data_for_aspect("MF")
    #print(go_pred_tableF1_MF)
    pred_mean_df_MF.T.to_csv(os.path.join(path,"text_representations/result_visualization/figures/pred_mean_df_MF.csv"))
    pred_mean_df_BP.T.to_csv(os.path.join(path,"text_representations/result_visualization/figures/pred_mean_df_BP.csv"))
    pred_mean_df_CC.T.to_csv(os.path.join(path,"text_representations/result_visualization/figures/pred_mean_df_CC.csv"))


    tables = {}
    tables["MF"] = go_pred_tableF1_MF
    tables["BP"] = go_pred_tableF1_BP
    tables["CC"] = go_pred_tableF1_CC
	
    pred_mean_df_MF.loc['F1-Weighted'].sort_values()
    pred_mean_df_BP.loc['F1-Weighted'].sort_values()
    pred_mean_df_CC.loc['F1-Weighted'].sort_values()


    g = sns.clustermap(go_pred_tableF1_MF, annot=True, cmap="YlGnBu", row_cluster=False,figsize=(15, 15))
    ax = g.ax_heatmap
    ax.set_xlabel("")
    ax.set_ylabel("")
    set_colors_and_marks_for_representation_groups(ax)
    g.savefig(os.path.join(path,'text_representations/result_visualization/figures/func_pred_MF.png'))


    g = sns.clustermap(go_pred_tableF1_BP, annot=True, cmap="YlGnBu", row_cluster=False,figsize=(15, 15))
    ax = g.ax_heatmap
    ax.set_xlabel("")
    ax.set_ylabel("")
    set_colors_and_marks_for_representation_groups(ax)
    g.savefig(os.path.join(path,'text_representations/result_visualization/figures/func_pred_BP.png'))

    g = sns.clustermap(go_pred_tableF1_CC, annot=True, cmap="YlGnBu", row_cluster=False,figsize=(15, 15))
    ax = g.ax_heatmap
    ax.set_xlabel("")
    ax.set_ylabel("")
    set_colors_and_marks_for_representation_groups(ax)
    g.savefig(os.path.join(path,'text_representations/result_visualization/figures/func_pred_CC.png'))



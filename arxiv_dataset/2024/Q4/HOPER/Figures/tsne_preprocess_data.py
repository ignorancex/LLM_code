import pandas as pd
import numpy as np
import pickle 

#Read propagated_go_terms_human.pkl file.
with open("/content/drive/MyDrive/propagated_go_terms_human.pkl", 'rb') as f:
    data = pickle.load(f)

#Filtered by GO_I and DB_OBJECT_ID
data_filter=data.filter(['GO_ID' , 'DB_OBJECT_ID'])
data_filter_reset_index=data_filter.reset_index()
data_filter_reset_index_x=data_filter_reset_index.filter(['GO_ID' , 'DB_OBJECT_ID'])
data_filter_reset_index_x_rename=data_filter_reset_index_x.rename({'DB_OBJECT_ID':'Entry'}, axis='columns')
data_filter_reset_index_x_rename.filter(['GO_ID']).drop_duplicates()

#Read dissimilar_terms_dataframe.pkl file.

dissimilar_df= pd.read_pickle("/content/drive/MyDrive/dissimilar_terms_dataframe.pkl")
dissimilar_df_filt=dissimilar_df.filter(['GO_ID'])
#Filtered according to the go ids found in the dissimilar_terms_dataframe.pkl file
GO_ID_data_propagated=pd.DataFrame()
lengthB=len(dissimilar_df_filt)
for i in range(lengthB):
 GO_ID_data_propagated=GO_ID_data_propagated.append(data_filter_reset_index_x_rename[data_filter_reset_index_x_rename["GO_ID"] == dissimilar_df_filt["GO_ID"][i]])
GO_ID_data_propagated_reset=GO_ID_data_propagated.reset_index()
GO_ID_data_propagated_reset_filt=GO_ID_data_propagated_reset.filter(['GO_ID' , 'Entry'])
GO_ID_data_propagated_reset_filt['GO_ID'].drop_duplicates()
#Read Simple_AE_512_dim_400_epochs.csv file.(different files for mf and bb,cc)
multi_col_representation_df=pd.read_csv('/content/drive/MyDrive/Simple_AE_512_dim_400_epochs.csv')

#Added GO_IDs of 'Entry'.
multi_col_representation_df_filt=multi_col_representation_df.join(GO_ID_data_propagated_reset_filt.set_index('Entry'), on='Entry')
multi_col_representation_df_filt_dropna=multi_col_representation_df_filt.dropna()
multi_col_representation_df_filt_dropna_dub=multi_col_representation_df_filt_dropna.drop_duplicates()
multi_col_representation_df_filt_dropna_dub.to_csv("Simple_AE_512_dim_400_epochs_go_id_drop_dublicates.csv")
go_category_dataframe = pd.read_pickle("/content/drive/MyDrive/go_category_dataframe (1).pkl")

#The naming for plot titles in the drawing has been changed.
go_category_dataframe['Aspect'] = go_category_dataframe['Aspect'] .str.replace('cellular_component','CC')
go_category_dataframe['Aspect'] = go_category_dataframe['Aspect'] .str.replace('biological_process','BP')
go_category_dataframe['Aspect'] = go_category_dataframe['Aspect'] .str.replace('molecular_function','MF')

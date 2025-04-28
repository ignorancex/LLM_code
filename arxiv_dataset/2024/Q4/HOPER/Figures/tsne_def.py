import pandas as pd
import numpy as np
import seaborn as sns
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from bioinfokit.visuz import cluster
from sklearn.manifold import TSNE
from sklearn import decomposition
import random

random_state=42
def aspect_num_cat_terms(go_category_dataframe,multi_col_representation_processdata,aspect,num_cat,termsif):
  go_category_dataframe_aspect=go_category_dataframe[go_category_dataframe.Aspect==aspect]
  go_category_dataframe_aspect_filtered= go_category_dataframe_aspect.loc[( go_category_dataframe_aspect['Number_Category'] == num_cat) & (go_category_dataframe_aspect['Term_Specificity'] == termsif)]
  Cat_Spec_ID=go_category_dataframe_aspect_filtered.GO_IDs.values
  Cat_Spec_df= pd.DataFrame()
  Cat_Spec_df ['GO_ID']= Cat_Spec_ID[0]
  GO_ID_data_Cat_Spec=pd.DataFrame()
  length_C_T=len(Cat_Spec_df)
  for i in range(length_C_T):
     GO_ID_data_Cat_Spec= GO_ID_data_Cat_Spec.append(multi_col_representation_processdata[multi_col_representation_processdata["GO_ID"] == Cat_Spec_df["GO_ID"][i]])
  multi_col_df_drop_dub=GO_ID_data_Cat_Spec.drop_duplicates()
  multi_col_df_drop_dub_res_in=multi_col_df_drop_dub.reset_index()
  multi_col_df_drop_dub_res_in_drop_index=multi_col_df_drop_dub_res_in.drop('index',axis=1)
  multi_col_df_drop_dub_res_in_drop_index_drop_ent=multi_col_df_drop_dub_res_in_drop_index.drop('Entry',axis=1)
  multi_col_df_drop_dub_res_in_drop_index_drop_ent_GO_ID= multi_col_df_drop_dub_res_in_drop_index_drop_ent.drop('GO_ID',axis=1)
  train= multi_col_df_drop_dub_res_in_drop_index_drop_ent_GO_ID
  labels= multi_col_df_drop_dub_res_in_drop_index_drop_ent['GO_ID']
  standarized_data = StandardScaler().fit_transform(train)
  sample_data=standarized_data
  model=TSNE(perplexity=10)
  tsne_data=model.fit_transform(sample_data)
  tsne_data=np.vstack((tsne_data.T,labels)).T
  tsne_df=pd.DataFrame(data=tsne_data,columns=("dim_1","dim_2","GO_ID"))
  plt.figure(figsize=(10,10))
  plot=sns.kdeplot(data=tsne_df, x="dim_1", y="dim_2", hue="GO_ID",alpha=0.50)
  plt.title(aspect+' '+num_cat+' '+termsif)
  plt.savefig("/content/drive/MyDrive/review_tsne"+aspect+'_'+num_cat+'_'+termsif+".png",dpi=(600.0))
  plt.show()
aspect=['MF']
num_cat=['Low','Middle','High']
termsif=['Shallow','Normal','Specific']

for i in termsif:
  for j in num_cat:
    for k in aspect:
      random.seed(random_state)
      np.random.seed(random_state)
      aspect_num_cat_terms(go_category_dataframe,multi_col_representation_processdata,k,j,i)


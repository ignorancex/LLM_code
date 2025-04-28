import pandas as pd
intactdata_dataframe=pd.read_excel('./ppi_representations/data/small_example.xlsx', engine='openpyxl')
intactdata_dataframe=intactdata_dataframe.drop(["Unnamed: 0"],axis=1)
#converted to edgelist format required for vector operation.
protein=[]
protein=intactdata_dataframe['#ID(s) interactor A']
interactionprotein=intactdata_dataframe['ID(s) interactor B']

list =[]
for col in intactdata_dataframe.columns:
    val = intactdata_dataframe[col]
    for v in val:
        list.append(v)

lengt=intactdata_dataframe.shape[0]
from collections import defaultdict
temp = defaultdict(lambda: len(temp))
res = [temp[ele] for ele in list]
P=res[0:lengt]
i=res[lengt:len(res)]
intactdata_dataframe= pd.DataFrame()
intactdata_dataframe['#ID(s) interactor A']=P
intactdata_dataframe['ID(s) interactor B']=i
print(intactdata_dataframe)
intactdata_dataframe.to_csv("./ppi_representations/data/example.edgelist",header=False, sep=' ', index=False)
#The order of the proteins corresponding to the numbers in the edgelist was recorded.

newlist =[]
for i in temp.keys():
    newlist.append(i)

protein_id_df=pd.DataFrame(newlist)
protein_id_df.to_csv("./ppi_representations/data/proteins_id.csv")


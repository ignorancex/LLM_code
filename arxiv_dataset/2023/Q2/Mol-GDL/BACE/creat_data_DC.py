import os
import numpy as np
import pandas as pd
from distance import mol_dis_sim
from feature import mol_feature
from utils_test import TestbedDataset
from splitters import ScaffoldSplitter

def get_element_index(mat, num):
    arr_index = []
    for i in range(len(mat)):
        _index = []
        for j in range(len(mat[i])):
            if float(mat[i][j]) == num:
                _index.append(i)
                _index.append(j)
                arr_index.append(_index)
                _index = []
    return arr_index

def get_new_coor(coor,Net_type):
    arr_Element = Net_type.split("-")
    new_coor = []
    for i in range(len(coor)):
        if coor[i].split("\t")[0] in arr_Element:
            new_coor.append(coor[i])
    return new_coor
    
def smile_to_graph3(arr_coor,List_cutoff):
    c_size = []
    features = []
    edge_indexs = []
    for i in range(len(List_cutoff)):
        arr_cutoff = List_cutoff[i].split("-")

        drug_dis, drug_dis_real, drug_atoms = mol_dis_sim.Calculate_distance(arr_coor,arr_cutoff)           
        drug_feat = mol_feature.Calculate_feature(drug_dis_real,drug_atoms)
        edge_index = get_element_index(drug_dis,1.0)
        
        c_size.append(len(arr_coor))
        features.append(drug_feat)
        edge_indexs.append(edge_index)
             
    return c_size, features, edge_indexs

def smile_to_graph4(arr_coor):
    drug_dis, drug_dis_real, drug_atoms = mol_dis_sim.Calculate_distance(arr_coor)            
    drug_feat = mol_feature.Calculate_feature(drug_dis_real,drug_atoms)
    edge_index = get_element_index(drug_dis,1.0)
    return [len(arr_coor)], [drug_feat], [edge_index]



def get_default_sider_task_names():
    """Get that default sider task names and return the side results for the drug"""

    return ['label']

def creat_data(datafile):

    
    fr_3d = open('3D_coor.txt','r')    
    List_cutoff = ['0-2','2-4','4-6','6-8','8-1000']
    
    smile_graph = {}
    arr_coor = []    
    for line in fr_3d:
        
        arrLin = line.strip().split("\t")
        if len(arrLin) == 4:
            arr_coor.append(line.strip())
        elif len(arrLin) == 1:
#            g = smile_to_graph4(arr_coor)
            g = smile_to_graph3(arr_coor,List_cutoff)
#            g = smile_to_graph2(arr_coor,List_cutoff,arrLin[0])
            smile_graph[arrLin[0]] = g
            arr_coor = []
            
    datasets = datafile
    processed_data_file_train = 'data/processed/' + datasets + '_train.pt'
#
    if ((not os.path.isfile(processed_data_file_train))):
        df = pd.read_csv('data/' + datasets + '.csv')#
        smiles_list, labels = df['smiles'], df[get_default_sider_task_names()]        
        labels = labels.replace(0, -1)
        
        data_list = []
        for i in range(len(smiles_list)):            
            data_list.append([smiles_list[i],labels.values[i],smile_graph[smiles_list[i]]])

        splitter = ScaffoldSplitter().split(data_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        
        train_smiles,train_label,train_graph = list(np.array(splitter[0])[:,0]), list(np.array(splitter[0])[:,1]), list(np.array(splitter[0])[:,2])
        valid_smiles,valid_label,valid_graph = list(np.array(splitter[1])[:,0]), list(np.array(splitter[1])[:,1]), list(np.array(splitter[1])[:,2])
        test_smiles,test_label,test_graph = list(np.array(splitter[2])[:,0]), list(np.array(splitter[2])[:,1]), list(np.array(splitter[2])[:,2])
        
        TestbedDataset(root='data', dataset = datafile + "_train", xd = train_smiles, y = train_label, smile_graph = train_graph)
        TestbedDataset(root='data', dataset = datafile + "_validation", xd = valid_smiles, y = valid_label, smile_graph = valid_graph)
        TestbedDataset(root='data', dataset = datafile + "_test", xd = test_smiles, y = test_label, smile_graph = test_graph)
        print('preparing ', datasets + '_.pt in pytorch format!')
    

if __name__ == "__main__":
    da = ['bace']
    for datafile in da:
        creat_data(datafile)

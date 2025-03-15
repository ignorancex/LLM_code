from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import deepchem as dc #for RDKit descriptors
import numpy as np

import pandas as pd

df_atom1 = pd.read_csv('../tables/descriptors/df_atom1_ext.csv').set_index("can")
df_global = pd.read_csv('../tables/descriptors/df_global_ext.csv').set_index("can")

data_Q_tot = {}

data_Q_tot['global']=df_global

data_Q_tot['atom1']=df_atom1

df_sigman = pd.read_excel('../tables/descriptors/sub_descriptors_sigman.xlsx', sheet_name = ['ortho','meta', 'para'])


def get_names(descriptors):
    
    if descriptors == 'rdkit':
        names = []
        for desc_name, function in Descriptors.descList :
            names.append(desc_name)
            
    elif descriptors == 'quantum':
        names = np.concatenate((data_Q_tot['global'].columns.to_list(), data_Q_tot['atom1'].columns.to_list()))
        
    elif descriptors == "hammett":
        names = list(np.concatenate((df_sigman['ortho'].set_index('R-ortho').columns.to_list(), df_sigman['meta'].set_index('R-meta').columns.to_list(), df_sigman['para'].set_index('R-para').columns.to_list())))
    
    return(names)


def B_index(mol):
    for i, at in enumerate(mol.getAtoms()):
        if at.symbol() == "B":
            N = i
    return N


def generate_morgan_matrix(smiles):
    morgan_matrix = np.zeros((1,1024))
    
    for smi in smiles : 
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
        fp = fp.ToBitString()        
        matrix_row = np.array([int(x) for x in list(fp)])
        morgan_matrix = np.row_stack((morgan_matrix, matrix_row))
    #print(morgan_matrix)
           
    #deleting first row of zeros
    morgan_matrix = np.delete(morgan_matrix, 0, axis=0)
    
    #print('\n')
    #print('dimensions :', morgan_matrix.shape)
    
    return(morgan_matrix)

def get_features(df, smi):
    """ df: featurized dataset
        smi: SMILES of the molecule
        returns: feature vector of the molecule"""    
    features = np.concatenate((np.array(df['global'].loc[smi]), np.array(df['atom1'].loc[smi])))
    return features

def create_descriptors(smiles, descriptors, data = data_Q_tot, structure = None, n_bit = 1024):
    """ smiles : list of smiles for the molecules
        descriptors: "fingerprints", "quantum", "rdkit", "hammett"; type : str
        returns : X: descriptors of the molecules """
    if descriptors == "fingerprints":
            X = generate_morgan_matrix(smiles)
    if descriptors == "quantum":
        X = []
        df = pd.Series(data) 
        for smi in smiles :
            X.append(get_features(df, smi))
        X = np.array(X)
    if descriptors == "rdkit":
        featurizer = dc.feat.RDKitDescriptors()
        X = featurizer.featurize(smiles) 
  
    if descriptors == "hammett": ## Substituents descriptors
        df = df_sigman
        X = []
        for smi in smiles :
            sub_ortho = id_substructure_ortho(smi, structure)
            features_ortho = np.array(df['ortho'].set_index('R-ortho').loc[sub_ortho])


            sub_meta = id_substructure_meta(smi, structure)
            features_meta = np.array(df['meta'].set_index('R-meta').loc[sub_meta])

            sub_para = id_substructure_para(smi, structure)
            features_para = np.array(df['para'].set_index('R-para').loc[sub_para])

            features = np.concatenate((features_ortho, features_meta, features_para))

            X.append(features)
        X = np.array(X)
    return(X)


def id_substructure_para(smi, structure):
    
    if structure == 'ONO' : 
        para_Br = Chem.MolFromSmarts('[#35]-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:1')
        para_N = Chem.MolFromSmarts('[#1]-[#7](-[#1])-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:1')
        para_OH = Chem.MolFromSmarts('[#1]-[#8]-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:1')
        para_Cl = Chem.MolFromSmarts('[#17]-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:1')
        para_F = Chem.MolFromSmarts('[#9]-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:1')
        para_Me = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:1')
        para_OMe = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#8]-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:1')
        para_NMe2 = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])(-[#1])(-[#1]))-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:1')
        para_CN = Chem.MolFromSmarts('[#7]#[#6]-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:1')
        para_tBu = Chem.MolFromSmarts('[#6]-[#6](-[#6])(-[#6])-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:1')
        para_CF3 = Chem.MolFromSmarts('[#9]-[#6](-[#9])(-[#9])-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:1')
        para_NO2 = Chem.MolFromSmarts('[#8]=[#7+](-[#8-])-[#6]1:[#6]:[#6]:[#6]2-[#8]-[#5]-[#7]-[#6]:2:[#6]:1')
      
    elif structure == 'triarylboranes':
        para_Br = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6](-[#35]):[#6]:[#6]:1')
        para_N = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6](-[#7](-[#1])-[#1]):[#6]:[#6]:1')
        para_OH = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6](-[#8]-[#1]):[#6]:[#6]:1')
        para_Cl = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6](-[#17]):[#6]:[#6]:1')
        para_F = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6](-[#9]):[#6]:[#6]:1')
        para_Me = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6](-[#6](-[#1])(-[#1])-[#1]):[#6]:[#6]:1')
        para_OMe = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]([#8]-[#6](-[#1])(-[#1])-[#1]):[#6]:[#6]:1')
        para_NMe2 = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6](-[#7](-[#6](-[H])(-[H])-[H])-[#6](-[H])(-[H])-[H]):[#6]:[#6]:1')
        para_CN = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6](-[#6]#[#7]):[#6]:[#6]:1')
        para_tBu = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6](-[#6](-[#6](-[H])(-[H])-[H])(-[#6](-[H])(-[H])-[H])-[#6](-[H])(-[H])-[H]):[#6]:[#6]:1')
        para_CF3 = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6](-[#6](-[#9])(-[#9])-[#9]):[#6]:[#6]:1')
        para_NO2 = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6](-[#7+](-[#8-])=[#8]):[#6]:[#6]:1')
    
    elif structure == 'NNN':
        para_Br = Chem.MolFromSmarts('[#35]-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:1')
        para_N = Chem.MolFromSmarts('[#1]-[#7](-[#1])-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:1')
        para_OH = Chem.MolFromSmarts('[#1]-[#8]-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:1')
        para_Cl = Chem.MolFromSmarts('[#17]-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:1')
        para_F = Chem.MolFromSmarts('[#9]-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:1')
        para_Me = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:1')
        para_OMe = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#8]-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:1')
        para_NMe2 = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])(-[#1])(-[#1]))-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:1')
        para_CN = Chem.MolFromSmarts('[#7]#[#6]-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:1')
        para_tBu = Chem.MolFromSmarts('[#6]-[#6](-[#6])(-[#6])-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:1')
        para_CF3 = Chem.MolFromSmarts('[#9]-[#6](-[#9])(-[#9])-[#6]1:[#6]:[#6]:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:1')
        para_NO2 = Chem.MolFromSmarts('[#8]=[#7+](-[#8-])-[#6]1:[#6]:[#6]:[#6]2-[#7](-[#1])-[#5]-[#7]-[#6]:2:[#6]:1')

    
    
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    if mol.HasSubstructMatch(para_Br):
        return('Br')
    elif mol.HasSubstructMatch(para_N):
        return('NH2')
    elif mol.HasSubstructMatch(para_OH):
        return('OH')
    elif mol.HasSubstructMatch(para_Cl):
        return('Cl')
    elif mol.HasSubstructMatch(para_F):
        return('F')
    elif mol.HasSubstructMatch(para_Me):
        return('Me')
    elif mol.HasSubstructMatch(para_OMe):
        return('OCH3')
    elif mol.HasSubstructMatch(para_NMe2):
        return('NMe2')
    elif mol.HasSubstructMatch(para_CN):
        return('CN')
    elif mol.HasSubstructMatch(para_tBu):
        return('tBu')
    elif mol.HasSubstructMatch(para_CF3):
        return('CF3')
    elif mol.HasSubstructMatch(para_NO2):
        return('NO2')
    else :
        return('H')
    
    

    
def id_substructure_meta(smi, structure):
    
    if structure == 'ONO':
        meta_Br = Chem.MolFromSmarts('[#35]-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#8]-[#6]:2:[#6]:1')
        meta_N = Chem.MolFromSmarts('[#1]-[#7](-[#1])-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#8]-[#6]:2:[#6]:1')
        meta_OH = Chem.MolFromSmarts('[#1]-[#8]-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#8]-[#6]:2:[#6]:1')
        meta_Cl = Chem.MolFromSmarts('[#17]-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#8]-[#6]:2:[#6]:1')
        meta_F = Chem.MolFromSmarts('[#9]-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#8]-[#6]:2:[#6]:1')
        meta_Me = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#8]-[#6]:2:[#6]:1')
        meta_OMe = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#8]-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#8]-[#6]:2:[#6]:1')
        meta_NMe2 = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])(-[#1])(-[#1]))-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#8]-[#6]:2:[#6]:1')
        meta_CN = Chem.MolFromSmarts('[#7]#[#6]-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#8]-[#6]:2:[#6]:1')
        meta_tBu = Chem.MolFromSmarts('[#6]-[#6](-[#6])(-[#6])-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#8]-[#6]:2:[#6]:1')
        meta_CF3 = Chem.MolFromSmarts( '[#9]-[#6](-[#9])(-[#9])-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#8]-[#6]:2:[#6]:1')
        meta_NO2 = Chem.MolFromSmarts('[#8-]-[#7+](-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#8]-[#6]:2:[#6]:1)=[#8]')
       
    elif structure == 'triarylboranes':
        
        meta_Br = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6](-[#35]):[#6]:1')
        meta_N = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6](-[#7](-[#1])-[#1]):[#6]:1')
        meta_OH = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6](-[#8]-[#1]):[#6]:1')
        meta_Cl = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6](-[#17]):[#6]:1')
        meta_F = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6](-[#9]):[#6]:1')
        meta_Me = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6](-[#6](-[#1])(-[#1])-[#1]):[#6]:1')
        meta_OMe = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6]([#8]-[#6](-[#1])(-[#1])-[#1]):[#6]:1')
        meta_NMe2 = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6](-[#7](-[#6](-[H])(-[H])-[H])-[#6](-[H])(-[H])-[H]):[#6]:1')
        meta_CN = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6](-[#6]#[#7]):[#6]:1')
        meta_tBu = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6](-[#6](-[#6](-[H])(-[H])-[H])(-[#6](-[H])(-[H])-[H])-[#6](-[H])(-[H])-[H]):[#6]:1')
        meta_CF3 = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6](-[#6](-[#9])(-[#9])-[#9]):[#6]:1')
        meta_NO2 = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6](-[#7+](-[#8-])=[#8]):[#6]:1')
    
    elif structure == 'NNN':
        meta_Br = Chem.MolFromSmarts('[#35]-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#7](-[#1])-[#6]:2:[#6]:1')
        meta_N = Chem.MolFromSmarts('[#1]-[#7](-[#1])-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#7](-[#1])-[#6]:2:[#6]:1')
        meta_OH = Chem.MolFromSmarts('[#1]-[#8]-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#7](-[#1])-[#6]:2:[#6]:1')
        meta_Cl = Chem.MolFromSmarts('[#17]-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#7](-[#1])-[#6]:2:[#6]:1')
        meta_F = Chem.MolFromSmarts('[#9]-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#7](-[#1])-[#6]:2:[#6]:1')
        meta_Me = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#7](-[#1])-[#6]:2:[#6]:1')
        meta_OMe = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#8]-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#7](-[#1])-[#6]:2:[#6]:1')
        meta_NMe2 = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])(-[#1])(-[#1]))-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#7](-[#1])-[#6]:2:[#6]:1')
        meta_CN = Chem.MolFromSmarts('[#7]#[#6]-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#7](-[#1])-[#6]:2:[#6]:1')
        meta_tBu = Chem.MolFromSmarts('[#6]-[#6](-[#6])(-[#6])-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#7](-[#1])-[#6]:2:[#6]:1')
        meta_CF3 = Chem.MolFromSmarts( '[#9]-[#6](-[#9])(-[#9])-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#7](-[#1])-[#6]:2:[#6]:1')
        meta_NO2 = Chem.MolFromSmarts('[#8-]-[#7+](-[#6]1:[#6]:[#6]:[#6]2-[#7]-[#5]-[#7](-[#1])-[#6]:2:[#6]:1)=[#8]')
           
        
    
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    if mol.HasSubstructMatch(meta_Br):
        return('Br')
    elif mol.HasSubstructMatch(meta_N):
        return('NH2')
    elif mol.HasSubstructMatch(meta_OH):
        return('OH')
    elif mol.HasSubstructMatch(meta_Cl):
        return('Cl')
    elif mol.HasSubstructMatch(meta_F):
        return('F')
    elif mol.HasSubstructMatch(meta_Me):
        return('Me')
    elif mol.HasSubstructMatch(meta_OMe):
        return('OCH3')
    elif mol.HasSubstructMatch(meta_NMe2):
        return('NMe2')
    elif mol.HasSubstructMatch(meta_CN):
        return('CN')
    elif mol.HasSubstructMatch(meta_tBu):
        return('tBu')
    elif mol.HasSubstructMatch(meta_CF3):
        return('CF3')
    elif mol.HasSubstructMatch(meta_NO2):
        return('NO2')
    else :
        return('H')
    
    
def id_substructure_ortho(smi, structure):
    
    if structure == 'ONO':
        ortho_Br = Chem.MolFromSmarts('[#35]-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:[#6]:[#6]:1')
        ortho_N = Chem.MolFromSmarts('[#1]-[#7](-[#1])-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:[#6]:[#6]:1')
        ortho_OH = Chem.MolFromSmarts('[#1]-[#8]-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:[#6]:[#6]:1')
        ortho_Cl = Chem.MolFromSmarts('[#17]-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:[#6]:[#6]:1')
        ortho_F = Chem.MolFromSmarts('[#9]-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:[#6]:[#6]:1')
        ortho_Me = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:[#6]:[#6]:1')
        ortho_OMe = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#8]-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:[#6]:[#6]:1')
        ortho_NMe2 = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])(-[#1])(-[#1]))-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:[#6]:[#6]:1')
        ortho_CN = Chem.MolFromSmarts('[#7]#[#6]-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:[#6]:[#6]:1')
        ortho_tBu = Chem.MolFromSmarts('[#6]-[#6](-[#6])(-[#6])-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:[#6]:[#6]:1')
        ortho_CF3 = Chem.MolFromSmarts('[#9]-[#6](-[#9])(-[#9])-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#8]-2):[#6]:[#6]:[#6]:1')
        ortho_NO2 = Chem.MolFromSmarts('[#6]1(:[#6]:[#6]:[#6]:[#6]2:[#6]:1-[#8]-[#5]-[#7]-2)-[#7+](=[#8])-[#8-]')
     
    elif structure == 'triarylboranes':
        ortho_Br = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#35]')
        ortho_N = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#7](-[#1])-[#1]')
        ortho_OH = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#8]-[#1]')
        ortho_Cl = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#17]')
        ortho_F = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#9]')
        ortho_Me = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#6](-[#1])(-[#1])-[#1]')
        ortho_OMe = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#8]-[#6](-[#1])(-[#1])-[#1]')
        ortho_NMe2 = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#7](-[#6](-[H])(-[H])-[H])-[#6](-[H])(-[H])-[H]')
        ortho_CN = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#6]#[#7]')
        ortho_tBu = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#6](-[#6](-[H])(-[H])-[H])(-[#6](-[H])(-[H])-[H])-[#6](-[H])(-[H])-[H]')
        ortho_CF3 = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#6](-[#9])(-[#9])-[#9]')
        ortho_NO2 = Chem.MolFromSmarts('[#5]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#7+](-[#8-])=[#8]')

        
    elif structure == 'NNN':
        ortho_Br = Chem.MolFromSmarts('[#35]-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:[#6]:[#6]:1')
        ortho_N = Chem.MolFromSmarts('[#1]-[#7](-[#1])-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:[#6]:[#6]:1')
        ortho_OH = Chem.MolFromSmarts('[#1]-[#8]-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:[#6]:[#6]:1')
        ortho_Cl = Chem.MolFromSmarts('[#17]-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:[#6]:[#6]:1')
        ortho_F = Chem.MolFromSmarts('[#9]-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:[#6]:[#6]:1')
        ortho_Me = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:[#6]:[#6]:1')
        ortho_OMe = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#8]-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:[#6]:[#6]:1')
        ortho_NMe2 = Chem.MolFromSmarts('[#1]-[#6](-[#1])(-[#1])-[#7](-[#6](-[#1])(-[#1])(-[#1]))-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:[#6]:[#6]:1')
        ortho_CN = Chem.MolFromSmarts('[#7]#[#6]-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:[#6]:[#6]:1')
        ortho_tBu = Chem.MolFromSmarts('[#6]-[#6](-[#6])(-[#6])-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:[#6]:[#6]:1')
        ortho_CF3 = Chem.MolFromSmarts('[#9]-[#6](-[#9])(-[#9])-[#6]1:[#6]2:[#6](-[#7]-[#5]-[#7](-[#1])-2):[#6]:[#6]:[#6]:1')
        ortho_NO2 = Chem.MolFromSmarts('[#6]1(:[#6]:[#6]:[#6]:[#6]2:[#6]:1-[#7](-[#1])-[#5]-[#7]-2)-[#7+](=[#8])-[#8-]')


         



    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    if mol.HasSubstructMatch(ortho_Br):
        return('Br')
    elif mol.HasSubstructMatch(ortho_N):
        return('NH2')
    elif mol.HasSubstructMatch(ortho_OH):
        return('OH')
    elif mol.HasSubstructMatch(ortho_Cl):
        return('Cl')
    elif mol.HasSubstructMatch(ortho_F):
        return('F')
    elif mol.HasSubstructMatch(ortho_Me):
        return('Me')
    elif mol.HasSubstructMatch(ortho_OMe):
        return('OCH3')
    elif mol.HasSubstructMatch(ortho_NMe2):
        return('NMe2')
    elif mol.HasSubstructMatch(ortho_CN):
        return('CN')
    elif mol.HasSubstructMatch(ortho_tBu):
        return('tBu')
    elif mol.HasSubstructMatch(ortho_CF3):
        return('CF3')
    elif mol.HasSubstructMatch(ortho_NO2):
        return('NO2')
    else :
        return('H')  
    









    
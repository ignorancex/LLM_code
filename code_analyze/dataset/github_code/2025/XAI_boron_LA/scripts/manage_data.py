import pandas as pd
import numpy as np



from rdkit import Chem


def replace_with_rdkit_smiles(df):
    for i in range(len(df)):
        smi = df.at[i,'SMILES']
        try : df.at[i,'SMILES'] = Chem.MolToSmiles(Chem.MolFromSmiles(df.at[i,'SMILES']))
        except : print(smi)
    return(df)

def create_df(csv_file):
    df = pd.read_csv(csv_file)
    df = replace_with_rdkit_smiles(df)
    df = df.set_index('SMILES')
    return(df)


def index_data(df):
    for i in range(len(df)):
        df.at[i,'can'] = Chem.MolToSmiles(Chem.MolFromSmiles(df.at[i,'can'])) #write RDKit SMILES
                                                                          
    df = df.drop_duplicates(subset='can') #remove duplicates from database
    df = df.set_index('can')
    return(df)


def rdkit_smi(smi):
    rdkit_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    return rdkit_smi

def list_rdkit_smi(iterable):
    l = []
    for smi in iterable :
        l.append(rdkit_smi(smi))
    return l

def draw_molecules(iterable_smi, legend = None):
    mols = []
    for smi in iterable_smi:
        mols.append(Chem.MolFromSmiles(smi))
    return(Chem.Draw.MolsToGridImage(mols, maxMols = len(mols), legends = legend))


def LA_only(data):
    ''' this fonction removes all structures containing [B-] substructure from quantum data'''
    df_g = data['global']
    df_at = data['atom1']
    smiles = df_g.index
    smiles_restricted = [smi for smi in smiles if '[B-]' not in smi]
    df_at = df_at[df_at.index.isin(smiles_restricted)]
    df_g = df_g[df_g.index.isin(smiles_restricted)]

    data= {'global' : df_g, 
           'atom1'  : df_at}
    return data
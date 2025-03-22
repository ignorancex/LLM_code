from rdkit import Chem
import numpy as np


def generate_boronics(N, smi_already_run, structure):
        
    """ smi_base = SMILES of the base structure with groups to be completed
        frags = list of SMILES for potential fragments to add
        N = number of SMILES to generate
        smi_already_run = list of already calculated SMILES
        returns a list of SMILES """
    
    frags = ['N', 'O', 'Cl', 'F', 'Br', 'C', 'OC', 'N(C)C', '[H]','[H]','[H]','[H]','[H]','[H]','[H]','[H]','[H]','[H]',"C#N", "C(C)(C)C", "C(F)(F)F","[N+]([O-])=O"]
    
    if  structure == 'ONO':
        smi_base = 'c1(*)c(*)c(OB4N2(c3c(O4)c(*)c(*)c(*)c3))c2cc1(*)'
        boronics_smi = []
        i = 0
        while i < N :    
            L = list(smi_base)
            L[3] = frags[np.random.randint(0, high=len(frags), size=None)]
            L[30]=L[3]
            L[7] = frags[np.random.randint(0, high=len(frags), size=None)]
            L[26] = L[7]
            L[34] = frags[np.random.randint(0, high=len(frags), size=None)]
            L[46] = L[34]
            indexes = []
            new_L = "".join(L)
            #print(new_L)
            mol  = Chem.MolFromSmiles(new_L)
            smi = Chem.MolToSmiles(mol)
            if (smi not in boronics_smi) and (smi not in smi_already_run):
                boronics_smi.append(smi)
                i += 1       
        return boronics_smi
   
    
    
    if structure == 'NNN':
        smi_base = 'c1(*)c(*)c(NB4N2(c3c(N4)c(*)c(*)c(*)c3))c2cc1(*)'
        boronics_smi = []
        i = 0
        while i < N :    
            L = list(smi_base)
            L[3] = frags[np.random.randint(0, high=len(frags), size=None)]
            L[30]=L[3]
            L[7] = frags[np.random.randint(0, high=len(frags), size=None)]
            L[26] = L[7]
            L[34] = frags[np.random.randint(0, high=len(frags), size=None)]
            L[46] = L[34]
            indexes = []
            new_L = "".join(L)
            #print(new_L)
            mol  = Chem.MolFromSmiles(new_L)
            smi = Chem.MolToSmiles(mol)
            if (smi not in boronics_smi) and (smi not in smi_already_run):
                boronics_smi.append(smi)
                i += 1       
        return boronics_smi
    
    if structure == 'OCO':
        smi_base = 'C1(*)=CC(c2c(O3)c(*)cc(*)c2)=C4B3Oc5c(*)cc(*)cc5C4=C1'
        boronics_smi = []
        i = 0
        while i < N :    
            L = list(smi_base)
            L[3] = frags[np.random.randint(0, high=len(frags), size=None)]
            L[18] = frags[np.random.randint(0, high=len(frags), size=None)]
            L[38]=L[18]
            L[23] = frags[np.random.randint(0, high=len(frags), size=None)]
            L[43] = L[23]            
            indexes = []
            new_L = "".join(L)
            #print(new_L)
            try : mol  = Chem.MolFromSmiles(new_L)
            except : print(new_L)
            try : smi = Chem.MolToSmiles(mol)
            except : print(new_L)
            if (smi not in boronics_smi) and (smi not in smi_already_run):
                boronics_smi.append(smi)
                i += 1
        return boronics_smi    
               
     
    if structure == 'triarylboranes':
        smi_base = 'C1(*)=C(*)C(*)=C(*)C(*)=C1B(C2=C(*)C(*)=C(*)C(*)=C2*)C3=C(*)C(*)=C(*)C(*)=C3*'
        boronics_smi = []
        p = 0
        while p < N :
            L = list(smi_base)
            #frag = frags[np.random.randint(0, high=len(frags), size=None)]
            indexes = [3, 21, 33, 51, 58, 76]
            for i in indexes :
                
                L[i] = '[H]'
            frag = frags[np.random.randint(0, high=len(frags), size=None)]
            indexes = [8, 17, 37, 46, 62, 71]
            for i in indexes :
                L[i] = frag
            frag = frags[np.random.randint(0, high=len(frags), size=None)]
            indexes = [12, 42, 67]
            for i in indexes :
                L[i] = frag
            
            new_L = "".join(L)
            #print(new_L)
            try : 
                mol  = Chem.MolFromSmiles(new_L)    
                smi = Chem.MolToSmiles(mol)               
                if (smi not in boronics_smi) and (smi not in smi_already_run):
                    boronics_smi.append(smi)
                    p += 1
            except :
                print(new_L)
                p = p 
        return boronics_smi



def get_fluoride_adduct(mol_list):
    mol_fluoride = []
    
    fluoride = Chem.MolFromSmiles('[F-]')
    for mol in mol_list:
        # combine mol and fluoride fragment
        combo = Chem.CombineMols(mol,fluoride)
        
        # get F- and B atom indexes
        for at in combo.GetAtoms():
            if at.GetSymbol() == 'B':
                num_B = at.GetIdx()
            elif at.GetSymbol() == 'F' and at.GetFormalCharge() == -1:
                num_F = at.GetIdx()
        
        # get an editable molecule and make the B-F bond
        edcombo = Chem.EditableMol(combo)
        edcombo.AddBond(num_F,num_B,order=Chem.rdchem.BondType.SINGLE)
        
        # get back the mol object and adjust formal charges
        mol = edcombo.GetMol()
        for at in mol.GetAtoms():
            if at.GetSymbol() == 'B':
                at.SetFormalCharge(-1)
            if at.GetSymbol() == 'F' and at.GetFormalCharge() == -1:
                at.SetFormalCharge(0)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol)) 
        mol_fluoride.append(mol)
    return mol_fluoride


def get_hydride_adduct(mol_list):
    mol_hydride = []
    
    hydride = Chem.MolFromSmiles('[H-]')
    for mol in mol_list:
        # combine mol and hydride fragment
        combo = Chem.CombineMols(mol,hydride)
        
        # get F- and B atom indexes
        for at in combo.GetAtoms():
            if at.GetSymbol() == 'B':
                num_B = at.GetIdx()
            elif at.GetSymbol() == 'H' and at.GetFormalCharge() == -1:
                num_H = at.GetIdx()
        
        # get an editable molecule and make the B-F bond
        edcombo = Chem.EditableMol(combo)
        edcombo.AddBond(num_H,num_B,order=Chem.rdchem.BondType.SINGLE)
        
        # get back the mol object and adjust formal charges
        mol = edcombo.GetMol()
        for at in mol.GetAtoms():
            if at.GetSymbol() == 'B':
                at.SetFormalCharge(-1)
            if at.GetSymbol() == 'H' and at.GetFormalCharge() == -1:
                at.SetFormalCharge(0)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol)) 
        mol_hydride.append(mol)
    return mol_hydride
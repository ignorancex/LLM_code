from autoqchem.db_functions import *
from autoqchem.slurm_manager import slurm_manager
from autoqchem.helper_classes import slurm_status
from autoqchem.helper_classes import slurm_job
from generate_smiles import *

import os
from rdkit import Chem

sm=slurm_manager(user='jfenogli', host='127.0.0.1', file_name="boron_LA")

data_directory = sm.workdir


def get_info_opt_freq(inchikey):
    pattern_CPU_time = " *Job cpu time: *"
    pattern_thermal_corr_to_H = " *Thermal correction to Enthalpy"
    patternSCF = ".*SCF\sDone:\s{1,}E\S{1,}\s=\s{1,}\S{1,}\s{1,}A.U.\safter\s{1,}\d{1,}\scycles.*"
    with open(f"{inchikey}_conf_0.log", 'r') as f:
        lines = f.readlines()    
    thermal_corr = []
    SCF = []
    for line in lines :        
        result_thermal_corr = re.match(pattern_thermal_corr_to_H, line)
        resultSCF = re.match(patternSCF, line)        
        if result_thermal_corr is not None :
            words = line.split()               
            value = float(words[-1])
            thermal_corr.append(value)
        if resultSCF is not None : 
            resultSCF = resultSCF.string
            value = float(find_float(resultSCF)[0])
            SCF.append(value)
    if thermal_corr == []:
        thermal_corr.append('no freq calc')
    return(thermal_corr[0], SCF[-1])


def find_float(string) : 
    L = re.findall("-{0,1}\d{1,}\.\d{1,}", string)
    return L


def create_nrj_dict(smiles):
    os.chdir(data_directory)
    dict_nrj_molecules = dict.fromkeys(smiles)
    keys = {'thermal_corr_to_H', 'SCF_nrj'}
    for smi in smiles :
        dict_nrj_molecules[smi] = dict.fromkeys(keys)
        job = sm.get_jobs(can = smi)
        inchikey = list(job.values())[0].inchikey
        if os.path.exists(inchikey):
            os.chdir(inchikey)
            if os.path.exists(f"{inchikey}_conf_0.log") :
                dict_nrj_molecules[smi]['thermal_corr_to_H'], dict_nrj_molecules[smi]['SCF_nrj'] = get_info_opt_freq(inchikey)
            os.chdir("../")
    return(dict_nrj_molecules)


#os.chdir(data_directory)
#dict_fluorine_donor = {'C[Si](C)(C)F' : None, 'C[Si+](C)C' : None}
#keys = {'thermal_corr_to_H', 'SCF_nrj'}
#for smi in dict_fluorine_donor.keys() :
#    dict_fluorine_donor[smi] = dict.fromkeys(keys)
 #   job = sm.get_jobs(can = smi)
 #   inchikey = list(job.values())[0].inchikey
 #   if os.path.exists(inchikey):
 #       os.chdir(inchikey)
 #       if os.path.exists(f"{inchikey}_conf_0.log") :
  #          dict_fluorine_donor[smi]['thermal_corr_to_H'], dict_fluorine_donor[smi]['SCF_nrj'] = get_info_opt_freq(inchikey)
   #     os.chdir("../")
dict_fluorine_donor = {'C[Si](C)(C)F': {'thermal_corr_to_H': 0.124589, 'SCF_nrj': -509.025634898}, 'C[Si+](C)C': {'thermal_corr_to_H': 0.116724, 'SCF_nrj': -408.859818742}}


def FIA_anchored(smi, smi_F, dict_nrj_molecules, dict_nrj_molecules_F) : #in kJ/mol
    
    correctedH_Me3_Si = (dict_fluorine_donor['C[Si+](C)C']['thermal_corr_to_H'] + dict_fluorine_donor['C[Si+](C)C']['SCF_nrj'])*2600
    correctedH_Me3_Si_F = (dict_fluorine_donor['C[Si](C)(C)F']['thermal_corr_to_H'] + dict_fluorine_donor['C[Si](C)(C)F']['SCF_nrj'])*2600
    #print(dict_nrj_molecules_F[smi_F]['thermal_corr_to_H'])
    try : 
        correctedH_mol_F = (dict_nrj_molecules_F[smi_F]['thermal_corr_to_H'] + dict_nrj_molecules_F[smi_F]['SCF_nrj'])*2600
        #print("correctedH_mol_F", correctedH_mol_F)
    except :
        FIA = "missing correctedH_mol_F"
        return(FIA)    
    try : 
        correctedH_mol = (dict_nrj_molecules[smi]['thermal_corr_to_H'] + dict_nrj_molecules[smi]['SCF_nrj'])*2600
        #print("correctedH_mol", correctedH_mol)
    except :
        FIA = "missing correctedH_mol"
        return(FIA)        
    FIA = -(correctedH_mol_F + correctedH_Me3_Si - (correctedH_mol + correctedH_Me3_Si_F + 952.5))
    #print("FIA", FIA)
    return(FIA)

def list_FIA(smiles):
    FIA = []
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    mols_F = get_fluoride_adduct(mols)
    smiles_F = [Chem.MolToSmiles(mol) for mol in mols_F]

    dict_nrj_molecules = create_nrj_dict(smiles)
    dict_nrj_molecules_F = create_nrj_dict(smiles_F)
    #print(dict_nrj_molecules_F)

    for i in range(len(smiles)):
        fia = FIA_anchored(smiles[i], smiles_F[i], dict_nrj_molecules, dict_nrj_molecules_F)
        FIA.append(fia)
    return(FIA)


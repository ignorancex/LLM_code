from moderna import *
import numpy as np
import os
import pickle
from tqdm import tqdm

def handle_rna_pdb(pdb_path = "result/PZ1.pdb"):
    atom_list = ['P', "O5'", "C5'", "C4'", "C3'", "O3'"]
    NAN_list = [float('nan'),float('nan'),float('nan')]
    with open(pdb_path, 'r') as file:
        coord_all = {}
        for atom_item in atom_list:
            coord_all[atom_item] = []
        
        chains = []
        seq = ''
        chain_idx = 0
        chain_idxs = []
        current = 0
        pointer = -1
        for i, line in enumerate(file):
            if line[0:4] != 'ATOM':
                chain_idx += 1
                continue
            chains.append(line[21])
            chain_idxs.append(chain_idx)
            #print(line[23:26].strip(),line[19])
            if int(line[23:26].strip()) != current:
                current = int(line[23:26].strip())
                pointer += 1
                seq+=line[19]
                for atom_item in atom_list:
                    coord_all[atom_item].append(NAN_list)

            coords_list = [line[30:37], line[38:45], line[46:53]]
            coords = []
            for coord_one in coords_list:
                if coord_one != '':
                    coords.append(float(coord_one))
            atom = line[11:16].strip()
            if atom in atom_list:
                coord_all[atom][pointer] = coords
            #print(atom)
            #coords = [float(line[31:37]), float(line[41:46]), float(line[46:54])]
            #print(coords)
        chains = list(set(chains))        
        num_chains = len(chains)
        name = pdb_path.split('/')[-1].split('.')[0]
        #print(seq, coord_all, chain_idxs, num_chains, name, chains)
    ### valid check
    
    for atom in atom_list:    
        if np.isnan(np.array(coord_all[atom])).all():
            return None
        #if atom in atom_list:
        #    coord_all[atom] = np.array(coord_all[atom])
    ss = ''
    for one_chain in chains:
        #print(one_chain)
        m = load_model(pdb_path, one_chain)
        ss += m.get_secstruc()

    final_dict = {'seq': seq, 'coords': coord_all, 'chain_idxs': chain_idxs, 'num_chains': num_chains, 'name': name, 'ss': ss, 'cluster': 0}
    return final_dict


#result_dir = "trRosettaRNA/20_RNA_Puzzles/pdb"
pdb_paths = []
result_dir = "RNA-puzzle/raw_dataset_and_for_assessment-master/result"
#result_dir = "trRosettaRNA/20_RNA_Puzzles/pdb"
for file_name in os.listdir(result_dir):
    if file_name.endswith(".pdb"):
        pdb_path = os.path.join(result_dir, file_name)
        pdb_paths.append(pdb_path)

rna_dataset = []
for pdb_path in tqdm(pdb_paths):
    result = handle_rna_pdb(pdb_path)
    if result is None:
        print("invalid sample, skip it.")
        continue
    
    rna_dataset.append(result)
with open('processed_pt/rna_puzzle.pt', 'wb') as file:
    pickle.dump(rna_dataset, file)
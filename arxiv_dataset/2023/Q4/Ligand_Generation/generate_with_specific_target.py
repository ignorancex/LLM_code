import os 
import argparse
import atom3d.util.formats as fo 
from tqdm import tqdm
import selfies as sf
import pandas as pd
import torch
from utils.protein_utils import featurize_as_graph
import os
import argparse
import random
import warnings
from collections import defaultdict
import torch
from torch.utils import data
from torch.utils.data import DataLoader

import numpy as np

from torch_geometric.loader import DataLoader
import copy

from models.pl_conditional_model import ThreeD_Conditional_VAE
from torch_geometric.data import Batch
import json
from multiprocessing import Pool
import multiprocessing
import subprocess

from tdc.chem_utils.oracle.oracle import smiles_to_rdkit_mol, qed, penalized_logp, jnk3, zaleplon_mpo, Vina_smiles, Vina_3d, SA
import warnings
import json
from get_esm import get_esm
import pandas as pd

from tdc.chem_utils import MolConvert

import time
import pickle
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import cairosvg
import io
converter = MolConvert(src = 'SMILES', dst = 'SELFIES')
inverter = MolConvert(src='SELFIES', dst = 'SMILES')


warnings.filterwarnings("ignore")

def molecule_to_pdf(mol, file_name, width=300, height=300):
    """Save substance structure as PDF"""
    # Render high resolution molecule
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Export to pdf
    cairosvg.svg2png(bytestring=drawer.GetDrawingText().encode(), write_to=file_name)

def remove_files(folder_path):
# List all files in the folder
    file_list = os.listdir(folder_path)
    # Loop through the files and remove them
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--target_idx", type = int, default=0)
parser.add_argument("--root", type = str, default = "./data/")
parser.add_argument("--protein_name", type = str, default = "1err")
parser.add_argument("--device", type = int, default = 0)
parser.add_argument("--num_epoch", type = int, default =200)
parser.add_argument("--chkpt_path", type = str, default = "./chkpt/")
parser.add_argument("--num_mols", type = int, default = 1000)
parser.add_argument("--num_generations", type = int, default = 20)
parser.add_argument('--prop', choices=['logp', 'penalized_logp', 'qed', 'sa', 'binding_affinity', 'multi_objective_binding_affinity'], default='multi_objective_binding_affinity')
parser.add_argument("--max_len", type=int, default = 30)
args = parser.parse_args()


num_mols = args.num_mols

root_dir = "./pdb"
protein_files = os.listdir("./pdb")
protein_pdbs = list(filter(lambda x: ".pdb" in x and ".pdbqt" not in x, protein_files))
receptor_files = list(filter(lambda x: "pdbqt" in x, protein_files))

# receptor_info_list = [
#     ('4r6e.pdb', -70.76, 21.82, 28.33, 15.0, 15.0, 15.0), 
#     ('3pbl.pdb', 9, 22.5, 26, 25, 25, 25), 
#     ('1iep.pdb', 15.6138918, 53.38013513, 15.454837, 80, 80, 80),
#     ('2rgp.pdb', 16.29212, 34.870818, 92.0353, 25, 25, 25),
#     ('3eml.pdb', -9.06363, -7.1446, 55.86259999, 25, 25, 25),
#     ('3ny8.pdb', 2.2488, 4.68495, 51.39820000000001, 25, 25, 25),
#     ('4rlu.pdb', -0.73599, 22.75547, -31.23689, 25, 25, 25),
#     ('4unn.pdb', 5.684346153, 18.1917, -7.3715, 25, 25, 25),
#     ('5mo4.pdb', -44.901, 20.490354, 8.48335, 25, 25, 25),
#     ('7l11.pdb', -21.81481, -4.21606, -27.98378, 25, 25, 25), ]

receptor_info_list = [
    ('4r6e.pdb', -70.76, 21.82, 28.33, 80.0, 80.0, 80.0), 
    ('3pbl.pdb', 9, 22.5, 26, 80, 80, 80), 
    ('1iep.pdb', 15.6138918, 53.38013513, 15.454837, 80, 80, 80),
    ('2rgp.pdb', 16.29212, 34.870818, 92.0353, 80, 80, 80),
    ('3eml.pdb', -9.06363, -7.1446, 55.86259999, 80, 80, 80),
    ('3ny8.pdb', 2.2488, 4.68495, 51.39820000000001, 80, 80, 80),
    ('4rlu.pdb', -0.73599, 22.75547, -31.23689, 80, 80, 80),
    ('4unn.pdb', 5.684346153, 18.1917, -7.3715, 80, 80, 80),
    ('5mo4.pdb', -44.901, 20.490354, 8.48335, 80, 80, 80),
    ('7l11.pdb', -21.81481, -4.21606, -27.98378, 80, 80, 80), ]
protein_pdbs.sort()
receptor_files.sort()
receptor_info_list = sorted(receptor_info_list, key = lambda x: x[0])

target_name = protein_pdbs[args.target_idx].replace(".pdb", "")
folder_name = f"ligands/{target_name}"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
remove_files(folder_name)

print("Target name: ", target_name)
print("Receptor file: ", receptor_files[args.target_idx])
print('Receptor info: ', receptor_info_list[args.target_idx])

print("Process target into graph and sequence feature")
protein_pdb = os.path.join(root_dir, protein_pdbs[args.target_idx])
protein_df = fo.bp_to_df(fo.read_pdb(protein_pdb))
protein_graph = featurize_as_graph(protein_df, device = "cuda")
esm_emb = get_esm(protein_pdb)

print("Get receptor information")
receptor_file_path = os.path.join(root_dir, receptor_files[args.target_idx])
receptor_info = receptor_info_list[args.target_idx]
_, center_x, center_y, center_z, size_x, size_y, size_z = receptor_info
func = Vina_3d(receptor_file_path, [center_x, center_y, center_z], [size_x, size_y, size_z])


model = ThreeD_Conditional_VAE(
    latent_dim = 128
)

model.load_state_dict(torch.load(f"checkpoint/target_aware.pt"))
model = model.to(device)
model.eval()

batch_mol = 500 if num_mols >= 1000 else num_mols
cnt = 0
smiles = []

while cnt < num_mols:
    print("Generate Batch: ", cnt)
    cond = [copy.deepcopy(protein_graph) for idx in range(batch_mol)]
    cond = Batch.from_data_list(cond).to(device)    
    seq = torch.stack([esm_emb for i in range(batch_mol)]).to(device)
     
    with torch.no_grad():
        x_prot = model.protein_model((cond.node_s, cond.node_v), 
                                        cond.edge_index, (cond.edge_s, cond.edge_v), seq, cond.batch)
        out = model.prior_network(x_prot).view(-1, 2, 128)
        prior_mu, prior_log_var = out[:, 0, :], out[:, 1, :]
        prior_std = torch.exp(prior_log_var * 0.5)
    z = torch.normal(mean = prior_mu, std = prior_std)
    cnt += batch_mol

outs = model.vae_model.sample(n_batch = z.shape[0], max_len = args.max_len, z = z)
smiles_list = []

for out in outs:
    try:
        smiles_list.append(inverter(out))
    except:
        continue


remove_files(folder_name)
for i, hot in enumerate(tqdm(smiles_list, desc='preparing ligands')):
    subprocess.Popen(f'obabel -:"{smiles_list[i]}" -O {folder_name}/{i}.pdbqt -p 7.4 --partialcharge gasteiger --gen3d', shell=True, stderr=subprocess.DEVNULL)
    time.sleep(2.0)

ligands = os.listdir(folder_name)
ligand_files = list(map(lambda x: os.path.join(folder_name, x), ligands))
output_files = list(map(lambda x: os.path.join(folder_name, f"out_{x}"), ligands))
indices = list(map(lambda x: int(x.replace(".pdbqt", "")), ligands))
items = list(zip(indices, ligand_files, output_files))
items = sorted(items, key = lambda x: x[0])
print("List for docking: ", items)
binding_score = list(map(lambda x: func(x[1], x[2]), items))
df_items = []
for idx, smiles in enumerate(smiles_list):
    df_items.append((idx, smiles, SA(smiles), qed(smiles), binding_score[idx]))
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    mol = Chem.MolFromSmiles(smiles)
    file_name = os.path.join(folder_name, f"mol_{idx}.png")
    molecule_to_pdf(mol, file_name)

df = pd.DataFrame(df_items, columns = ["index", "Smiles", "SA", "QED", "Binding Score"])
df.to_csv(f"{folder_name}/scores.csv", index = False)
print("Target name", target_name)
print(df.head(10))

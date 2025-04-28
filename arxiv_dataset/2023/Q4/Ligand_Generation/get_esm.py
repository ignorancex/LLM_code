from Bio.PDB import PDBParser
import torch
import esm
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning


#Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
model= model.to("cuda")

biopython_parser = PDBParser()

three_to_one = {'ALA':	'A',
'ARG':	'R',
'ASN':	'N',
'ASP':	'D',
'CYS':	'C',
'GLN':	'Q',
'GLU':	'E',
'GLY':	'G',
'HIS':	'H',
'ILE':	'I',
'LEU':	'L',
'LYS':	'K',
'MET':	'M',
'MSE':  'M', # this is almost the same AA as MET. The sulfur is just replaced by Selen
'PHE':	'F',
'PRO':	'P',
'PYL':	'O',
'SER':	'S',
'SEC':	'U',
'THR':	'T',
'TRP':	'W',
'TYR':	'Y',
'VAL':	'V',
'ASX':	'B',
'GLX':	'Z',
'XAA':	'X',
'XLE':	'J'}

def get_sequence(rec_path):
    # structure = biopython_parser.get_structure('random_id', rec_path)
    # structure = structure[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', rec_path)
        rec = structure[0]
    seq = ''  
    for i, chain in enumerate(rec):     
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid and not
                try:
                    seq += three_to_one[residue.get_resname()]
                except Exception as e:
                    seq += '-'
                    print("encountered unknown AA: ", residue.get_resname(), ' in the complex ', rec_path, '. Replacing it with a dash - .')
    return seq

def get_esm(pdb_path):
    seq = get_sequence(pdb_path)
    complex_name = None
    seq_data = [(complex_name, seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(seq_data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        batch_tokens = batch_tokens.to("cuda")
        results = model(batch_tokens, repr_layers = [33], return_contacts = False)
    token_representations = results["representations"][33]
    residue_embeddings = token_representations[:, 1 : batch_lens[0] - 1, :].squeeze()
    return torch.mean(residue_embeddings, dim =0)

def get_esm_from_sequence(seq):
    complex_name = None
    seq_data = [(complex_name, seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(seq_data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        batch_tokens = batch_tokens.to("cuda")
        results = model(batch_tokens, repr_layers = [33], return_contacts = False)
    token_representations = results["representations"][33]
    residue_embeddings = token_representations[:, 1 : batch_lens[0] - 1, :].squeeze()
    return torch.mean(residue_embeddings, dim =0)

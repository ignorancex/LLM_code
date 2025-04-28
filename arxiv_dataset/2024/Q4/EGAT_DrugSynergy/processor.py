import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from rdkit import Chem
import torch
from motifs import GenMotif
from rdkit.Chem import AllChem

class Processor_drug:

    def __init__(self, dataset, gene_expressions):
        self.dataset = dataset
        self.Expressions = gene_expressions

        self.symbols = ['C', 'N', 'O', 'F', "Cl", 'S', "Br", 'I', 'P', 'Se', 'Na', 'B', 'Si',
                        'K', "Zn", "Pt", "As", "Gd", "Li", "Sb", "Mg", "Mn", "Au", "Co"]

        self.electronegativities = {'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, 'Cl': 3.16, 'S': 2.58, 'Br': 2.96,
                                    'I': 2.66, 'P': 2.19, 'Se': 2.55, 'Na': 0.93, 'B': 2.04, 'Si': 1.9, "K": 0.82,
                                    "Zn": 1.65, "Pt": 2.28, "As": 2.18, "Gd": 1.2, "Li": 0.98, "Sb": 2.05,
                                    "Mg": 1.31, "Mn": 1.55, "Au": 2.54, "Co": 1.88}

        self.radii = {'C': .67, 'N': .56, 'O': .48, 'F': .42, 'Cl': .79, 'S': .87, 'Br': .94,
                      'I': 1.15, 'P': .98, 'Se': 1.03, 'Na': 1.9, 'B': .87, 'Si': 1.11, "K": 2.43,
                      "Zn": 1.42, "Pt": 1.77, "As": 1.14, "Gd": 2.33, "Li": 1.67, "Sb": 1.33,
                      "Mg": 1.45, "Mn": 1.61, "Au": 1.65, "Co": 1.52}

        self.hybridizations = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'other']

        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]

    def to_graph(self, smile):
        mol = Chem.MolFromSmiles(smile)
        new_molecule = []
        mol_coords = []
        for i, atom in enumerate(mol.GetAtoms()):
            symbol = [0.] * len(self.symbols)
            symbol[self.symbols.index(atom.GetSymbol())] = 1.
            degree = [0.] * 5
            degree[atom.GetDegree()] = 1.
            formal_charge = atom.GetFormalCharge()
            radical_electrons = atom.GetNumRadicalElectrons()
            hybridization = [0.] * len(self.hybridizations)
            hybridization[self.hybridizations.index(atom.GetHybridization())] = 1.
            hydrogens = [0.] * 5
            hydrogens[atom.GetTotalNumHs()] = 1.
            chirality_type = [0.] * 2
            if atom.HasProp('_CIPCode'):
                chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.
            aromaticity = 1. if atom.GetIsAromatic() else 0.
            chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
            electronegativity = self.electronegativities[atom.GetSymbol()]
            radius = self.radii[atom.GetSymbol()]
            x = torch.tensor(symbol + degree + [formal_charge] +
                             [radical_electrons] + hybridization + [aromaticity] + hydrogens + [chirality] +
                             chirality_type + [electronegativity] + [radius])
            new_molecule.append(x)

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        mol = Chem.RemoveHs(mol)

        for atom in mol.GetConformers()[0].GetPositions():
            coords = np.array([atom[0], atom[1], atom[2]])
            mol_coords.append(torch.tensor(np.sqrt(np.linalg.norm(np.sum(coords**2, axis=0)))))

        coords = torch.stack(mol_coords, dim=0)
        features = torch.stack(new_molecule, dim=0)

        edge_indices = []
        edge_attrs = []

        for bond in mol.GetBonds():
            edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
            conjugation = 1. if bond.GetIsConjugated() else 0.
            ring = 1. if bond.IsInRing() else 0.
            stereo = [0.] * 4
            edge_attr = torch.tensor(
                [single, double, triple, aromatic, conjugation, ring] + stereo)

            edge_attrs += [edge_attr, edge_attr]

        if len(edge_attrs) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices).t().contiguous()
            edge_attr = torch.stack(edge_attrs, dim=0)

        graph = Data(x=torch.Tensor(features),
                     edge_index=edge_index,
                     edge_attr=edge_attr,
                     coord=torch.reshape(torch.Tensor(coords), (torch.Tensor(features).shape[0], 1)),
                     smile=smile)

        return graph

    def process_index(self, index):
        drug1 = self.dataset["Drug1"][index]
        drug2 = self.dataset["Drug2"][index]
        CCL = self.dataset["Cell line"][index]

        try:
            Bliss = torch.from_numpy(np.array(self.dataset["Bliss"][index]).astype("float32"))
            if Bliss <= -10:
                Synergy = torch.tensor(0, dtype=torch.float32)
            elif Bliss >= 10:
                Synergy = torch.tensor(1, dtype=torch.float32)
            else:
                return None, None

            DrugGraph1 = self.to_graph(drug1)
            DrugGraph2 = self.to_graph(drug2)

            Expression = torch.from_numpy(self.Expressions[CCL].to_numpy().astype("float32"))
            HSA = torch.from_numpy(np.array(self.dataset["HSA"][index]).astype("float32"))
            Loewe = torch.from_numpy(np.array(self.dataset["Loewe"][index]).astype("float32"))
            ZIP = torch.from_numpy(np.array(self.dataset["ZIP"][index]).astype("float32"))

            return DrugGraph1, Data(x=DrugGraph2.x, edge_index=DrugGraph2.edge_index,
                                    edge_attr=DrugGraph2.edge_attr, coord=DrugGraph2.coord, smile=DrugGraph2.smile,
                                    Expression=Expression, ZIP=ZIP, Loewe=Loewe, HSA=HSA, Bliss=Bliss, Synergy=Synergy)
        except:
            return None, None

    def Process_Dataset(self):
        drugs1 = []
        drugs2 = []
        for index in tqdm(self.dataset.index):
            Drug1, Drug2 = self.process_index(index)
            if Drug1 and Drug2:
                drugs1.append(Drug1)
                drugs2.append(Drug2)
        return drugs1, drugs2

class Processor_motif:

    def __init__(self, dataset):
        self.dataset = dataset
        self.motifs = GenMotif()
        self.generate_motif_vocab()

    def generate_motif_vocab(self):
        for smile in tqdm(self.dataset[0].smile):
            self.motifs.update_MotifVocab(smile)
        keys = self.motifs.get_MotifVocab()
        emb = self.motifs.get_vocab_embettings()

    def Process_Dataset(self):
        motifs1 = []
        for smile in tqdm(self.dataset[0].smile):
            motifs1.append(self.motifs.generate_motifGraph(smile))
        return motifs1
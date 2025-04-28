import torch
from torch_geometric.data import Data
from rdkit import Chem
import numpy as np

class GenMotif(object):
    def __init__(self):
        self.vocab = {}  # SMILE String and its frequency
        self.atoms = ["C", "N", "O", "S", "CL", "F", "PT", "B", "P", "BR",
                      "SE"]  # Atoms not included in the cluster vocab
        self.vocab_embeddings = {}

    def create_clusters(self, smile):
        mol = Chem.MolFromSmiles(smile)

        clusters = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing() and (
                    not mol.GetAtomWithIdx(a1).IsInRing() and not mol.GetAtomWithIdx(a2).IsInRing()):
                clusters.append((a1, a2))

        ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
        clusters.extend(ssr)

        clusters_final = []
        skipped = []
        clusters = list(clusters)

        for cluster in clusters:
            if cluster in skipped:
                continue
            elif len(cluster) > 2:  # The cluster is a ring
                clusters_final.append(cluster)
            else:   # The cluster is not a ring
                new_cluster = list(cluster)
                for atoms in clusters:  # Make a new cluster with all of attached elements not in a ring
                    if len(atoms) == 2 and (atoms[0] in new_cluster or atoms[1] in new_cluster):
                        if atoms[0] not in new_cluster:
                            new_cluster.append(atoms[0])
                        if atoms[1] not in new_cluster:
                            new_cluster.append(atoms[1])
                        skipped.append(atoms)
                clusters_final.append(new_cluster)

        # Clusters_final is a list of clusters with each cluster having the atom indexes

        return clusters_final

    def update_MotifVocab(self, smile):
        # Creates a dictionary of SMILE motifs and their frequency
        mol = Chem.MolFromSmiles(smile)
        clusters = self.create_clusters(smile)

        for cluster in clusters:
            cluster_smile = Chem.MolFragmentToSmiles(mol, cluster)

            for index in cluster:
                atom = Chem.MolFragmentToSmiles(mol, [index]).upper()
                if atom not in self.atoms:
                    self.atoms.append(atom)

            if cluster_smile.upper() in self.vocab.keys():
                self.vocab[cluster_smile.upper()] += 1
            else:
                self.vocab.update({cluster_smile.upper(): 1})

    def get_MotifVocab(self):
        # Gets final motif vocabulary and deletes those with small frequency and adds to the list of atoms not represented by a motif
        for i in list(self.vocab.keys()):  # Goes through all the smile strings
            if self.vocab[i] < 100:  # Determines if it has high enough frequency
                del self.vocab[i]
        return list(self.vocab.keys())

    def get_vocab_embettings(self):
        keys = self.get_MotifVocab()
        keys.extend(self.atoms)  # Gets list of motifs and elements not represented by a motif
        for i in range(len(keys)):  # Creates one hot encoded tensor for each SMILE or element
            embedding = torch.from_numpy(np.zeros(len(keys)+1))
            embedding[i] = 1
            self.vocab_embeddings.update({keys[i]: embedding})

        embedding = torch.from_numpy(np.zeros(len(keys) + 1))
        embedding[-1] = 1

        self.vocab_embeddings.update({"other": embedding})
        return self.vocab_embeddings

    def get_neighboor(self, clusters1, cluster2, mol, encoding):
        clusters1 = clusters1 if isinstance(clusters1, list) else [clusters1]
        cluster2 = cluster2 if isinstance(cluster2, list) else [cluster2]

        for atom_index1 in clusters1:
            neighboors = [i.GetIdx() for i in mol.GetAtomWithIdx(atom_index1).GetNeighbors()]
            for atom_index2 in cluster2:
                if atom_index2 in neighboors:
                    return [[encoding[str(clusters1) if isinstance(clusters1, int) else ', '.join([str(j) for j in clusters1])],
                             encoding[str(cluster2) if isinstance(cluster2, int) else ', '.join([str(j) for j in cluster2])]]] \
                        + [[encoding[str(cluster2) if isinstance(cluster2, int) else ', '.join([str(j) for j in cluster2])],
                            encoding[str(clusters1) if isinstance(clusters1, int) else ', '.join([str(j) for j in clusters1])]]]
        return None
    def generate_motifGraph(self, smile):
        mol = Chem.MolFromSmiles(smile)
        clusters = [list(i) for i in self.create_clusters(smile)]
        new_clusters = []

        for i in range(len(clusters)):
            if Chem.MolFragmentToSmiles(mol, clusters[i]) not in self.vocab:  # Adds the molecule index as its own element (not represented by a motif)
                for index in clusters[i]:
                    new_clusters.append(index)
            else:  # Adds motif to the list
                new_clusters.append(clusters[i])

        edge_index = []
        node_features = []
        encoding = {}

        for i in range(len(new_clusters)):
            encoding.update({str(new_clusters[i]) if isinstance(new_clusters[i], int) else ', '.join([str(j) for j in new_clusters[i]]): i})

        for cluster_index1 in range(len(new_clusters)):
            for cluster in new_clusters[cluster_index1+1:]:
                edge = self.get_neighboor(new_clusters[cluster_index1], cluster, mol, encoding)
                if edge:
                    edge_index += edge
            node_features.append(self.vocab_embeddings[Chem.MolFragmentToSmiles(mol, new_clusters[cluster_index1] if isinstance(new_clusters[cluster_index1], list) else [new_clusters[cluster_index1]]).upper()])

        edge_index = torch.tensor(edge_index).t().contiguous()
        node_features = torch.stack(node_features, dim=0)
        return Data(x=torch.Tensor(node_features),
                    edge_index=edge_index)





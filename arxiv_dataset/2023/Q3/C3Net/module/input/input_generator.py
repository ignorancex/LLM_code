import multiprocessing
import os
from functools import partial

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol

# from module.input.element import solv_prop_nomalized
import module.input.element as element
import module.input.embedding as embedding
import module.input.feature as feature
import module.input.surface as surface
from module.input.vector import *


class Information:
    def __init__(self, lig, surface, emb):
        self.lig = lig
        self.surf = surface
        self.emb = emb
        self.encoder = []

        # bonded
        self.neighbor_index_bond = []
        self.neighbor_type_bond = []
        self.mask_bond = []

        # grid-atom
        self.neighbor_index_nonbond = []
        self.neighbor_distance_nonbond = []
        self.mask_nonbond = []

    def GetBondType(self, atom1, atom2, bond):
        bType = -1
        if bond == Chem.rdchem.BondType.AROMATIC:
            bType = 13
        else:
            hyb1 = str(atom1.GetHybridization())
            hyb2 = str(atom2.GetHybridization())
            if bond == Chem.rdchem.BondType.SINGLE:
                if atom1.GetSymbol() == "H":
                    if hyb2 == "SP3":
                        bType = 1
                    elif hyb2 == "SP2":
                        bType = 2
                    elif hyb2 == "SP":
                        bType = 3

                elif atom2.GetSymbol() == "H":
                    if hyb1 == "SP3":
                        bType = 1
                    elif hyb1 == "SP2":
                        bType = 2
                    elif hyb1 == "SP":
                        bType = 3

                elif hyb1 == "SP3":
                    if hyb2 == "SP3":
                        bType = 4
                    elif hyb2 == "SP2":
                        bType = 5
                    elif hyb2 == "SP":
                        bType = 6

                elif hyb1 == "SP2":
                    if hyb2 == "SP3":
                        bType = 5
                    elif hyb2 == "SP2":
                        bType = 7
                    elif hyb2 == "SP":
                        bType = 8
                    # print (bType)

                elif hyb1 == "SP":
                    if hyb2 == "SP3":
                        bType = 6
                    elif hyb2 == "SP2":
                        bType = 8
                    elif hyb2 == "SP":
                        bType = 8  # very rare, not defined

            elif bond == Chem.rdchem.BondType.DOUBLE:
                if hyb1 == "SP3":
                    bType = 9
                elif hyb1 == "SP2":
                    if hyb2 == "SP3":
                        bType = 9
                    elif hyb2 == "SP2":
                        bType = 10
                    elif hyb2 == "SP":
                        bType = 11
                elif hyb1 == "SP":
                    bType = 11

            elif bond == Chem.rdchem.BondType.TRIPLE:
                bType = 12

        return bType

    def AddBond(self, id1, id2, bond):
        self.neighbor_index_bond[id1].append(id2)
        self.neighbor_type_bond[id1].append(bond)

        self.neighbor_index_bond[id2].append(id1)
        self.neighbor_type_bond[id2].append(bond)

    def Bond(self):
        bond2type = {}
        patt = Chem.MolFromSmarts("[CX3](=[O,S])[NX3]")  # amide
        if self.lig.rdMol.HasSubstructMatch(patt):
            matches = self.lig.rdMol.GetSubstructMatches(patt)
            for match in matches:
                key = [match[0], match[1]]
                key.sort()
                bond2type[tuple(key)] = 14

                key = [match[0], match[2]]
                key.sort()
                bond2type[tuple(key)] = 14

        patt = Chem.MolFromSmarts("C(=O)[O-]")  # carboxlate
        if self.lig.rdMol.HasSubstructMatch(patt):
            matches = self.lig.rdMol.GetSubstructMatches(patt)
            for match in matches:
                key = [match[0], match[1]]
                key.sort()
                bond2type[tuple(key)] = 15

                key = [match[0], match[2]]
                key.sort()
                bond2type[tuple(key)] = 15

        patt = Chem.MolFromSmarts("[N+](=O)[O-]")  # nitro
        if self.lig.rdMol.HasSubstructMatch(patt):
            matches = self.lig.rdMol.GetSubstructMatches(patt)
            for match in matches:
                key = [match[0], match[1]]
                key.sort()
                bond2type[tuple(key)] = 16

                key = [match[0], match[2]]
                key.sort()
                bond2type[tuple(key)] = 16

        patt = Chem.MolFromSmarts("[NHX3][CH0X3](=[NH2X3+])[NH2X3]")  # guaidium
        if self.lig.rdMol.HasSubstructMatch(patt):
            matches = self.lig.rdMol.GetSubstructMatches(patt)
            for match in matches:
                key = [match[1], match[0]]
                key.sort()
                bond2type[tuple(key)] = 17

                key = [match[1], match[2]]
                key.sort()
                bond2type[tuple(key)] = 17

                key = [match[1], match[3]]
                key.sort()
                bond2type[tuple(key)] = 17

        for i in range(self.lig.atomN):
            atom = self.lig.rdMol.GetAtomWithIdx(i)
            batoms = atom.GetNeighbors()
            for batom in batoms:
                j = batom.GetIdx()
                if i < j:
                    if (i, j) in bond2type:
                        bType = bond2type[(i, j)]
                    else:
                        bond = self.lig.rdMol.GetBondBetweenAtoms(i, j).GetBondType()
                        bType = self.GetBondType(atom, batom, bond)
                    if bType == -1:  # in correct structures
                        return 0
                    self.AddBond(i, j, bType)
        return 1

    # distance between a grid point and ligand atoms
    def NonBond(self, position):
        for i in range(self.lig.atomN):
            for j in range(self.lig.atomN, len(position)):
                dis = distance(position[i], position[j])
                self.neighbor_index_nonbond[i].append(j)
                self.neighbor_distance_nonbond[i].append(dis / 8)

    def SetInformation(self):
        position = []
        types = []
        for i in range(self.lig.atomN):
            position.append(self.lig.coors[i])
            types.append(self.lig.atom_types[i])
            self.neighbor_index_bond.append([])
            self.neighbor_type_bond.append([])
            self.neighbor_index_nonbond.append([])
            self.neighbor_distance_nonbond.append([])

        for i in range(len(self.surf)):
            position.append(self.surf[i])

        if not self.Bond():
            return 0
        self.NonBond(position)

        self.mask_bond = self.Padding2D(
            [self.neighbor_index_bond, self.neighbor_type_bond]
        )
        self.mask_nonbond = self.Padding2D(
            [self.neighbor_index_nonbond, self.neighbor_distance_nonbond]
        )
        self.encoder = self.emb.embeddings(torch.tensor(types)).tolist()

        return 1

    def Padding2D(self, lists):
        maxL = 0

        for l in lists[0]:  # max number of neighbors (interaction partners)
            if len(l) > maxL:
                maxL = len(l)

        mask = np.ones([len(lists[0]), maxL], dtype="i")  # [obj num, max neighbor num]

        for i in range(len(lists[0])):
            l = len(lists[0][i])
            for j in range(l, maxL):
                for k in range(len(lists)):
                    lists[k][i].append(0.0)
                mask[i][j] = 0

        return np.array(mask, dtype="f")


def write_tfrecord(surfacePoints, lig, dG, solvent_prop, solvent_id, dir, emb):
    info = Information(lig, surfacePoints, emb)
    if info.SetInformation() == 0:
        return 1

    conf_index = int(lig.rdMol.GetProp("Conf_ID"))
    mol_index = int(lig.rdMol.GetProp("ID"))
    solute_id = int("%s%s" % (mol_index, conf_index))
    data = {}
    data["Encoder"] = np.array(info.encoder, dtype="f")
    data["Neighbor_Index_Bond"] = np.array(info.neighbor_index_bond, dtype="i")
    data["Neighbor_Index_NonBond"] = np.array(info.neighbor_index_nonbond, dtype="i")
    data["Neighbor_Type_Bond"] = np.array(info.neighbor_type_bond, dtype="i")
    data["Neighbor_Distance_NonBond"] = np.array(
        info.neighbor_distance_nonbond, dtype="f"
    )
    data["Mask_Bond"] = np.array(info.mask_bond, dtype="f")
    data["Mask_NonBond"] = np.array(info.mask_nonbond, dtype="f")
    data["Atom_Num"] = np.array([lig.atomN], dtype="i")
    data["Atom_Grid_Num"] = np.array([len(info.encoder)], dtype="i")
    data["Grid_Num"] = np.array([len(surfacePoints)], dtype="i")
    data["dG"] = np.array([dG], dtype="f")
    data["Solvent_ID"] = np.array([solvent_id], dtype="i")
    data["Solute_ID"] = np.array([solute_id], dtype="i")
    data["Solvent_Properties"] = np.array([solvent_prop], dtype="f")

    np.savez(f"{dir}/{solvent_id}_{solute_id}", **data)


def get_feature(m):
    f = feature.Feature(m)
    f.set_feature()
    return f


def GetSurface(f, spherePoints):
    surf = surface.Surface()
    return surf.molecule(f, spherePoints)


def check_atom(m):
    nitro = []
    patt = Chem.MolFromSmarts("[N+](=O)[O-]")  # nitro
    if m.HasSubstructMatch(patt):
        matches = m.GetSubstructMatches(patt)
        for match in matches:
            nitro.append(match[0])
            nitro.append(match[2])

    for i in range(m.GetNumAtoms()):
        symbol = m.GetAtomWithIdx(i).GetSymbol()
        charge = m.GetAtomWithIdx(i).GetFormalCharge()
        if symbol not in ["H", "C", "O", "N", "S", "P", "F", "Cl", "Br", "I"]:
            return 0
        if charge != 0 and i not in nitro:  # only neutral
            return 1
    return 1


def Run(sp, emb, solvent_prop, solvent_id, dir, m):
    # print (solvent_id)
    if check_atom(m):
        if not m.HasProp("Salt_Solvent") or m.GetProp("Salt_Solvent") == "?":
            if type(m) != type(None):
                f = get_feature(m)
                if type(f.atom_types) != type(None):
                    if m.HasProp("dG"):
                        dG = float(m.GetProp("dG"))
                    elif m.HasProp("logPapp"):
                        dG = float(m.GetProp("logPapp"))
                    else:
                        dG = 0.0
                    sp = GetSurface(f, sp)
                    if sp:
                        write_tfrecord(sp, f, dG, solvent_prop, solvent_id, dir, emb)
    else:
        print(m.GetProp("ID"), "ionic")


def GenerateConformer_RDKit():
    sdfs = [""]  # list of sdf file names to predict
    pool = multiprocessing.Pool(processes=len(os.sched_getaffinity(0)))
    pool.map(Run_GenerateConformer_RDKit, sdfs)
    pool.close()
    pool.join()


def Run_GenerateConformer_RDKit(f):
    from rdkit.Chem import AllChem

    supplier = Chem.SDMolSupplier(f, True, False)
    writer = Chem.SDWriter("%s_Conf5_RDKit.sdf" % f[:-4])
    for m in supplier:
        try:
            Chem.AssignStereochemistryFrom3D(m)
            AllChem.EmbedMultipleConfs(m, numConfs=50)
            AllChem.UFFOptimizeMoleculeConfs(m)
            nH_m = Chem.RemoveHs(m)

            AllChem.AlignMolConformers(nH_m)
            confs = m.GetConformers()

            selected = [0]
            for i in range(1, len(confs)):
                flag = True
                for j in selected:
                    rms = AllChem.GetConformerRMS(nH_m, i, j, prealigned=True)
                    if rms < 0.5:
                        flag = False
                        break
                if flag:
                    selected.append(i)
                if len(selected) >= 5:
                    break

            count = 0
            for i in selected:
                m.SetProp("Conf_ID", "%s" % count)
                writer.write(m, confId=confs[i].GetId())
                count += 1

        except:
            m.SetProp("Conf_ID", "%s" % 0)
            writer.write(m)


def generator_predict(solv: str, solu: str, dir: str):
    device = torch.device("cpu")
    solv = solv.lower()
    solvents = list(element.solv_prop_nomalized.keys())
    solv_id = solvents.index(solv)
    # print (solv, solv_id)
    prop = element.solv_prop_nomalized[solv]

    emb = embedding.Type2Vec()
    emb.load_state_dict(torch.load("../module/input/embedding.pt", map_location=device))

    sphere = surface.Sphere()
    spherePoints = sphere.draw()

    ligs = Chem.SDMolSupplier(solu, True, False)
    pms = []
    for i, m in enumerate(ligs):
        m.SetProp("ID", "%s" % i)
        m.SetProp("Conf_ID", str(0))
        pms.append(PropertyMol(m))

    pool = multiprocessing.Pool(processes=len(os.sched_getaffinity(0)))
    func = partial(Run, spherePoints, emb, prop, solv_id, dir)
    pool.map(func, pms)
    pool.close()
    pool.join()


def generator_train(dir):
    device = torch.device("cpu")

    emb = embedding.Type2Vec()
    emb.load_state_dict(torch.load("../module/input/embedding.pt", map_location=device))

    sphere = surface.Sphere()
    spherePoints = sphere.draw()

    solvents = list(element.solv_prop_nomalized.keys())

    for i, solvent in enumerate(solvents):
        if os.path.isfile(
            "../dataset/training_set/sdf/%s_Solute_Prep_Conf5_RDKit.sdf" % solvent
        ):
            print("solvent", i, solvent)

            prop = element.solv_prop_nomalized[solvent]
            ligs = Chem.SDMolSupplier(
                "../dataset/training_set/sdf/%s_Solute_Prep_Conf5_RDKit.sdf" % solvent,
                True,
                False,
            )
            Run(spherePoints, emb, prop, i, ligs[0])

            pms = [PropertyMol(lig) for lig in ligs]
            pool = multiprocessing.Pool(processes=len(os.sched_getaffinity(0)))
            func = partial(Run, spherePoints, emb, prop, i, dir)
            pool.map(func, pms)
            pool.close()
            pool.join()


if __name__ == "__main__":
    device = torch.device("cpu")

    emb = embedding.Type2Vec()
    emb.load_state_dict(torch.load("embedding.pt", map_location=device))

    sphere = surface.Sphere()
    spherePoints = sphere.draw()

    solvents = list(element.solv_prop_nomalized.keys())

    for i, solvent in enumerate(solvents):
        if os.path.isfile(
            "../../dataset/training_set/sdf/%s_Solute_Prep_Conf5_RDKit.sdf" % solvent
        ):
            print("solvent", i, solvent)

            prop = element.solv_prop_nomalized[solvent]
            ligs = Chem.SDMolSupplier(
                "../../dataset/training_set/sdf/%s_Solute_Prep_Conf5_RDKit.sdf"
                % solvent,
                True,
                False,
            )
            Run(spherePoints, emb, prop, i, ligs[0])

            pms = [PropertyMol(lig) for lig in ligs]
            pool = multiprocessing.Pool(processes=len(os.sched_getaffinity(0)))
            func = partial(Run, spherePoints, emb, prop, i)
            pool.map(func, pms)
            pool.close()
            pool.join()

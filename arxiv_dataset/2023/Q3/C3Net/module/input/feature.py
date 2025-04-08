#! /usr/bin/python
from module.input.molecule import *
import module.input.vdw_type as vdw

class Feature:
    def __init__(self,mol):
        self.rdMol = mol
        self.atomN = mol.GetNumAtoms()
        self.coors = []
        self.atom_types = []
        self.vdw_types = []
        self.atom_range = []

    def set_feature(self):
        m = Molecule(self.rdMol)
        self.atom_types = m.get_atom_type()
        self.vdw_types = vdw.get_vdw_type(self.rdMol)
        conf = self.rdMol.GetConformer()
        for i in range(self.atomN):
            self.atom_range.append(0)
            self.coors.append(conf.GetAtomPosition(i))





    

    


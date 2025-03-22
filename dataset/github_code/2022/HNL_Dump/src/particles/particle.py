import numpy as np

class Particle:
    def __init__(self, m, beam=None, parent=None, momenta=[]):
        self.m = m
        self.beam = beam
        self.parent = parent
        self.momenta = momenta
        self.children = []

    def boost(self, parent_momenta):
        length = min(len(self.momenta), len(parent_momenta))
        lab_momenta = []
        for i in range(length):
            p_parent = parent_momenta[i]
            p_child_rest = self.momenta[i]
            p_child_lab = p_child_rest.boost(-p_parent)
            lab_momenta.append(p_child_lab)
        self.momenta = lab_momenta
        return self
    
    def set_momenta(self, momenta):
        self.momenta = momenta
        return self
    
    def get_energies(self):
        if len(self.momenta) == 0:
            raise Exception("Momenta array is empty!")
        
        energies = []
        for p4 in self.momenta:
            energies.append(p4.get_energy())
        return np.array(energies)

    def get_transverse_masses(self):
        if len(self.momenta) == 0:
            raise Exception("Momenta array is empty!")
        
        transverse_masses = []
        for p4 in self.momenta:
            transverse_masses.append(p4.get_transverse_mass())
        return np.array(transverse_masses)
    
    def get_cos_thetas(self):
        if len(self.momenta) == 0:
            raise Exception("Momenta array is empty!")

        cos_thetas = []
        for p4 in self.momenta:
            cos_thetas.append(p4.get_cos_theta())
        return np.array(cos_thetas)

    def geometric_cut(self, min_angle, max_angle):
        max_cos_theta = np.cos(min_angle)
        min_cos_theta = np.cos(max_angle)

        cut_momenta = []
        for p in self.momenta:
            cos_theta = p.get_cos_theta()
            if cos_theta > min_cos_theta and cos_theta < max_cos_theta:
                cut_momenta.append(p)
        
        acceptance =  len(cut_momenta)/len(self.momenta)
        # replace the momenta with the cut momenta since we are only intersted in these from now on
        self.momenta = cut_momenta
        return acceptance
    
    def decay(self):
        return self

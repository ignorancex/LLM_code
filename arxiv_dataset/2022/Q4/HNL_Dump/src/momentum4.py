import numpy as np

class Momentum4:
    def __init__(self, p, m):
        self.p = p
        self.m = m

    @classmethod
    def from_cartesian(cls, e, px, py, pz, m):
        p = np.array([e, px, py, pz])
        return cls(p, m)
        
    @classmethod
    def from_polar(cls, e, c_theta, phi, m):
        p = np.array([e, np.sqrt(e**2 - m**2)*np.sqrt(1 - c_theta**2)*np.cos(phi),
            np.sqrt(e**2 - m**2)*np.sqrt(1 - c_theta**2)*np.sin(phi), np.sqrt(e**2 - m**2)*c_theta])
        return cls(p, m)
    
    @classmethod
    def from_e_pt_pp(cls, e, pt, pp, phi, m):
        p = np.array([e, pt*np.cos(phi), pt*np.sin(phi), pp])
        return cls(p, m)

    def boost(self, p4):
        p = p4.p
        gamma = p[0]/p4.m
        bx = p[1]/p[0]
        by = p[2]/p[0]
        bz = p[3]/p[0]
        b2 = bx*bx + by*by + bz*bz
        lor = np.array([[gamma, -gamma*bx, -gamma*by, -gamma*bz],
                        [-gamma*bx, 1 + (gamma - 1)*bx*bx/b2, (gamma - 1)*bx*by/b2, (gamma - 1)*bx*bz/b2],
                        [-gamma*by, (gamma - 1)*by*bx/b2, 1 + (gamma - 1)*by*by/b2, (gamma - 1)*by*bz/b2],
                        [-gamma*bz, (gamma - 1)*bz*bx/b2, (gamma - 1)*bz*by/b2, 1 + (gamma - 1)*bz*bz/b2]])
        return Momentum4(np.matmul(lor, self.p), self.m)
    
    def __neg__(self):
        p = np.array([self.p[0], -self.p[1], -self.p[2], -self.p[3]])
        return Momentum4(p, self.m)
    
    def __add__(self, other):
        p1 = np.array(self.p)
        p2 = np.array(other.p)
        p = p1 + p2
        m = np.sqrt(p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2) # invariant mass
        return Momentum4(p, m)
    
    def get_energy(self):
        return self.p[0]
    
    def get_cos_theta(self):
        return self.p[3]/np.sqrt(self.p[0]**2 - self.m**2)
    
    def get_transverse_momentum(self):
        return np.sqrt(self.p[1]**2 + self.p[2]**2)

    def get_parallel_momentum(self):
        return self.p[3]
    
    def get_total_momentum(self):
        return np.sqrt(self.p[1]**2 + self.p[2]**2 + self.p[3]**2)
    
    def get_transverse_mass(self):
        pt = self.get_transverse_momentum()
        return (np.sqrt(pt**2 + self.m**2) + pt)

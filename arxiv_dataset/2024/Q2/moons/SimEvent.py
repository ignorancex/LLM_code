import numpy as np

import unhash
import globs

class SimEvent:
    
    def __init__(self,typ,n,t):
        
        self.type = typ
        self.names = n
        self.t = t
        
        return
            
class CollisionEvent(SimEvent):
    
    def __init__(self,typ,n,t,sa):
        
        super().__init__(typ,n,t)
        
#        names = []
        indices = []
        
        for i,s in enumerate(sa):
            
#            names.append([unhash.unhash(p.hash,globs.glob_names) for p in s.particles])
            names = [unhash.unhash(p.hash,globs.glob_names) for p in s.particles]
            if n[0] in names and n[1] in names:
                indices.append(i)
                
        s = sa[max(indices)]   # final snapshot with both bodies
        p1 = s.particles[n[0]]
        p2 = s.particles[n[1]]
        
#        d = np.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)
#        print('restored coll at distance ',d)

        self.orb = p1.calculate_orbit(primary=p2)
        self.a = self.orb.a
        self.q = self.orb.a * (1-self.orb.e)

        return
        
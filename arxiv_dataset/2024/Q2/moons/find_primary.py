import numpy as np
import rebound

import unhash

def find_primary(part,planets,name_all,sim):
# user can feed the index, name or particle object itself
        
    try:
        if isinstance(part,int):
            p = sim.particles[part]
        if isinstance(part,str):
            p = sim.particles[part]
        if isinstance(part,rebound.Particle):
            p = part
            
# return None if body was removed
    except:
        return None
    
# we will use the rH of all the "planets" to determine primary
# actually we will compute rH based on orbital distance d not a as the semimajor axis can become very large...
    pls = []
    for pl in planets:
        try:
            if isinstance(pl,int):
                pls.append(sim.particles[pl])
            if isinstance(pl,str):
                pls.append(sim.particles[pl])
            if isinstance(pl,rebound.Particle):
                pls.append(pl)
        except rebound.ParticleNotFound:
            continue
            
    
    d2min = np.inf
    primary = unhash.unhash(sim.particles[0].hash,name_all)
# default to orbiting primary   
    for pl in pls:
        try:
            if pl.hash.value == p.hash.value:
                continue # don't want to be bound to oneself
            rh = pl.d * (pl.m/(3*sim.particles[name_all[0]].m))**(1/3)
            d2 = (p.x-pl.x)**2 + (p.y-pl.y)**2 + (p.z-pl.z)**2
            if d2 < rh**2 and d2 < d2min:
                d2min = d2
                primary = unhash.unhash(pl.hash,name_all)
        except:
            continue
            
    if primary == unhash.unhash(sim.particles[0].hash,name_all):
        orb = p.calculate_orbit(primary=sim.particles[0])
        if orb.a < 0:
            primary = 'Galaxy'
            
    return primary
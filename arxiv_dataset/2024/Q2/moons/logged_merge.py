import unhash
import globs
import numpy as np

def logged_merge(sim_pointer,collided_particles_index):
        
    sim = sim_pointer.contents # retrieve the standard simulation object
    ps = sim.particles # easy access to list of particles

    # here we check whether the merging bodies are both planets. If so, we need 
    # an additional check on whether they've really come close enough for a merge
    
    i = collided_particles_index.p1   # Note that p1 < p2 is not guaranteed.    
    j = collided_particles_index.p2 

    n1 = unhash.unhash(ps[i].hash,globs.glob_names)
    n2 = unhash.unhash(ps[j].hash,globs.glob_names)
    if 'Planet' in n1 and 'Planet' in n2:
        xx = (ps[i].x - ps[j].x)**2
        yy = (ps[i].y - ps[j].y)**2
        zz = (ps[i].z - ps[j].z)**2
        d2 = xx + yy + zz
        dcrit2 = (globs.glob_Rpl[0]+globs.glob_Rpl[1])**2
        if d2 > dcrit2:
#            print(f'Close passage of {n1} and {n2} with dist {np.sqrt(d2)} au at {sim.t} years')
            return 0
    
# save at this point
    sim.simulationarchive_snapshot(globs.glob_archive)
    
    # Merging Logic 
    total_mass = ps[i].m + ps[j].m
    merged_planet = (ps[i] * ps[i].m + ps[j] * ps[j].m)/total_mass # conservation of momentum

    # merged radius assuming keep the same mean density
    merged_radius = (total_mass**2/(ps[i].m**2/ps[i].r**3 + ps[j].m**2/ps[j].r**3))**(1/3)

#    print('A: Current bodies: '+' '.join([unhash.unhash(p.hash,globs.glob_names) for p in sim.particles]))
#    print('A: Current hashes: '+' '.join(str([p.hash for p in sim.particles])))


    # keep the more massive body and remove the lesser
    if ps[i].m >= ps[j].m:
        keep = i
        kill = j
    else:
        keep = j
        kill = i

#    print('B: Current bodies: '+' '.join([unhash.unhash(p.hash,globs.glob_names) for p in sim.particles]))
#    print('B: Current hashes: '+' '.join(str([p.hash for p in sim.particles])))

    nkill = unhash.unhash(ps[kill].hash,globs.glob_names)
    nkeep = unhash.unhash(ps[keep].hash,globs.glob_names)
#    print(keep,nkeep,kill,nkill)
    if nkill in globs.glob_planets and nkeep in globs.glob_planets:
        print(f'CE between {nkill} '
              f'and {nkeep} with dist '
              f'{globs.glob_darr[i-1][j-1][0]} au at {sim.t} years')
    print(f'{nkill} removed by collision with '
          f'{nkeep} at {sim.t} years')
    print(ps[nkill])
    print(ps[nkeep])
    orbit = ps[nkill].calculate_orbit(primary=ps[nkeep])
    print(f'Orbelts: a = {orbit.a} au; e = {orbit.e} ; I = {orbit.inc} ; Omega = {orbit.Omega}; '
          f'omega = {orbit.omega} ; MA = {orbit.M}')
    with open(globs.glob_log,'a') as f:
        if nkill in globs.glob_planets and nkeep in globs.glob_planets:
            print(f'CE between {nkill} '
                  f'and {nkeep} with dist '
                  f'{globs.glob_darr[i-1][j-1][0]} au at {sim.t} years',file=f)
        print(f'{nkill} removed by collision with '
              f'{nkeep} at {sim.t} years',file=f)
        print(ps[nkill],file=f)
        print(ps[nkeep],file=f)
        print(f'Orbelts: a = {orbit.a} au; e = {orbit.e} ; I = {orbit.inc} ; Omega = {orbit.Omega}; '
              f'omega = {orbit.omega} ; MA = {orbit.M}',file=f)
    
#    print('C: Current bodies: '+' '.join([unhash.unhash(p.hash,globs.glob_names) for p in sim.particles]))
#    print('C: Current hashes: '+' '.join(str([p.hash for p in sim.particles])))

    Ein = sim.calculate_energy()
    
    tmp = ps[keep].hash
    ps[keep] = merged_planet   # update p1's state vector (mass and radius will need corrections)
    ps[keep].m = total_mass    # update to total mass
    ps[keep].r = merged_radius # update to joined radius
    ps[keep].hash = tmp        # needed to avoid names f'ing up
# We have to remove the body here to get the energy logging correct, and then tell
# REBOUND NOT to remove something as an effect of the function return value
#    print('D: Current bodies: '+' '.join([unhash.unhash(p.hash,globs.glob_names) for p in sim.particles]))
#    print('D: Current hashes: '+' '.join(str([p.hash for p in sim.particles])))
#    print('Removing '+nkill)

    if nkill in globs.glob_planets:
        globs.glob_planets.remove(nkill)
        globs.glob_npl = globs.glob_npl-1
    sim.remove(kill)

    print('Current planets: '+' '.join(globs.glob_planets))
#    print('E: Current bodies: '+' '.join([unhash.unhash(p.hash,globs.glob_names) for p in sim.particles]))
#    print('E: Current hashes: '+' '.join(str([p.hash for p in sim.particles])))
    with open(globs.glob_log,'a') as f:
        print('Current planets: '+' '.join(globs.glob_planets),file=f)
    
    Eout = sim.calculate_energy()
    
#    if sim.track_energy_offset:
    sim.energy_offset += (Ein-Eout)
        
    return 0
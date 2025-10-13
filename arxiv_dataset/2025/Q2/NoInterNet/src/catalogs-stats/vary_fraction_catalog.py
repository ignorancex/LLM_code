import numpy as np
import argparse
import os

import readfof

import displace_diff_z as ddz

class FoF_catalog_contaminated:
    def __init__(self, TotNgroups, TotNids, GroupLen, GroupOffset, GroupMass, GroupPos, GroupVel, GroupTLen, GroupTMass, GroupType):
        
        dt1 = np.dtype((np.float32,3))
        dt2 = np.dtype((np.float32,6))
        
        self.Ngroups    = np.asarray(TotNgroups, dtype=np.int32)
        self.TotNgroups = np.asarray(TotNgroups, dtype=np.int32)
        self.Nids       = np.asarray(TotNids, dtype=np.int32)
        self.TotNids    = np.asarray(TotNids, dtype=np.uint64)

        self.GroupLen    = GroupLen.astype(dtype=np.int32)
        self.GroupOffset = GroupOffset.astype(dtype=np.int32)
        self.GroupMass   = GroupMass.astype(dtype=np.float32)
        self.GroupPos    = GroupPos.astype(dtype=np.float32)
        self.GroupVel    = GroupVel.astype(dtype=np.float32)
        self.GroupTLen   = GroupTLen.astype(dtype=np.float32)
        self.GroupTMass  = GroupTMass.astype(dtype=np.float32)
        self.GroupType   = GroupType.astype(dtype=np.int32)

def read_halo(snapdir, snapnum=2):

    FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=False,
                          swap=False, SFR=False, read_IDs=False)

    return FoF

def displace_inbox(FoF, f, axis=2, dz=100.0, BoxSize=1000.0, seed=7):
    """Inbox displacement of the entire FoF catalogue"""
    p = np.copy(FoF.GroupPos)/1e3
    
    np.random.seed(seed)
    ind = np.random.randint(0, len(p) - 1, int(len(p) * f))

    #select interlopers
    pi = p[ind,:]
    #TotNgroupst, TotNids, 
    GroupLent = np.delete(FoF.GroupLen, ind, axis=0)
    GroupLeni = FoF.GroupLen[ind]
    GroupOffsett = np.delete(FoF.GroupOffset, ind, axis=0)
    GroupOffseti = FoF.GroupOffset[ind]
    GroupMasst = np.delete(FoF.GroupMass, ind, axis=0)
    GroupMassi = FoF.GroupMass[ind]
    GroupPost = np.delete(FoF.GroupPos, ind, axis=0)
    GroupPosi = FoF.GroupPos[ind]
    GroupVelt = np.delete(FoF.GroupVel, ind, axis=0)
    GroupVeli = FoF.GroupVel[ind]
    GroupTLent = np.delete(FoF.GroupTLen, ind, axis=0)
    GroupTLeni = FoF.GroupTLen[ind]
    GroupTMasst = np.delete(FoF.GroupTLen, ind, axis=0)
    GroupTMassi = FoF.GroupTLen[ind]
    
    pi[:, axis] += dz
    if dz > 0:
        packman = pi[:,axis] > BoxSize
        pi[packman,axis] -= BoxSize
    elif dz < 0:
        packman = pi[:,axis] < 0
        pi[packman,axis] += BoxSize
    to_keep = (pi[:,2] < BoxSize) & (pi[:,2] > 0.) #this is needed if dz > BoxSize, we are removing object that remains out after packman
    GroupPosi = pi[to_keep]*1e3#GroupPosi[to_keep]
    GroupLeni = GroupLeni[to_keep]
    GroupOffseti = GroupOffseti[to_keep]
    GroupMassi = GroupMassi[to_keep]
    GroupVeli = GroupVeli[to_keep]
    GroupTLeni = GroupTLeni[to_keep]
    GroupTMassi = GroupTMassi[to_keep]
    
    GroupType = np.concatenate([np.zeros(len(GroupPost), dtype=np.uint32), np.ones(len(GroupPosi), dtype=np.uint32)])
    GroupPos = np.vstack((GroupPost, GroupPosi))
    GroupLen = np.hstack((GroupLent, GroupLeni))
    GroupOffset = np.hstack((GroupOffsett, GroupOffseti))
    GroupMass = np.hstack((GroupMasst, GroupMassi))
    GroupVel = np.vstack((GroupVelt, GroupVeli))
    GroupTLen = np.vstack((GroupTLent, GroupTLeni))
    GroupTMass = np.vstack((GroupTMasst, GroupTMassi))

    FoF_c = FoF_catalog_contaminated(len(GroupLen), len(GroupLen), GroupLen, GroupOffset, GroupMass, GroupPos, GroupVel, GroupTLen, GroupTMass, GroupType) #ToTNids not clear

    return FoF_c

def displace_outbox(FoFz1, FoFz2, f, separation, thetax, thetay, thetaz, axis=2, dz=1000., BoxSize=1000., seed=7):
    """Outbox displacement of the entire FoF catalogue"""
    np.random.seed(seed)

    pz1 = np.copy(FoFz1.GroupPos)/1e3
    pz2 = np.copy(FoFz2.GroupPos)/1e3

    #the rotation has to be done as first thing (for the velocities when I do it is not important)
    pz2 = ddz.rotatexyz(pz2, thetax, thetay, thetaz)

    pz2[:,axis] += separation
    N1, N2 = len(pz1), len(pz2)
    fr = f * N1 / (1. - f) / N2 #in this way the fraction of interlopers in the final sample is f
    ind = np.random.randint(0, len(pz2) - 1, int(len(pz2) * fr))
    
    p = pz2[ind]
    GroupLeni = FoFz2.GroupLen[ind]
    GroupOffseti = FoFz2.GroupOffset[ind]
    GroupMassi = FoFz2.GroupMass[ind]
    GroupVeli = FoFz2.GroupVel[ind]
    GroupTLeni = FoFz2.GroupTLen[ind]
    GroupTMassi = FoFz2.GroupTLen[ind]

    p[:,axis] += dz

    #no packman in this case they are displaced inside the box or discarded
    to_keep = (p[:,axis] < BoxSize) & (p[:,axis] > 0.)
    p = p[to_keep]
    GroupLeni = GroupLeni[to_keep]
    GroupOffseti = GroupOffseti[to_keep]
    GroupMassi = GroupMassi[to_keep]
    GroupVeli = GroupVeli[to_keep]
    GroupTLeni = GroupTLeni[to_keep]
    GroupTMassi = GroupTMassi[to_keep]

    #rotation of the positions and velocities of the interlopers
    v_rot = ddz.rotatexyz(GroupVeli, thetax, thetay, thetaz, center=np.array([0,0,0], dtype=np.float32))

    GroupType = np.concatenate([np.zeros(len(pz1), dtype=np.uint32), np.ones(len(p), dtype=np.uint32)])
    GroupPos = np.vstack((FoFz1.GroupPos, p*1e3))
    GroupLen = np.hstack((FoFz1.GroupLen, GroupLeni))
    GroupOffset = np.hstack((FoFz1.GroupOffset, GroupOffseti))
    GroupMass = np.hstack((FoFz1.GroupMass, GroupMassi))
    GroupVel = np.vstack((FoFz1.GroupVel, v_rot))
    GroupTLen = np.vstack((FoFz1.GroupTLen, GroupTLeni))
    GroupTMass = np.vstack((FoFz1.GroupTMass, GroupTMassi))
  
    FoF_c = FoF_catalog_contaminated(len(GroupLen), len(GroupLen), GroupLen, GroupOffset, GroupMass, GroupPos, GroupVel, GroupTLen, GroupTMass, GroupType) #ToTNids not clear

    return FoF_c

def main(ns):
    fractions = np.loadtxt(ns.f)
    fractions_check = ns.last_fs

    first, last = ns.first, ns.first + ns.number
    for i in range(first, last):
        snapdir = os.path.join(ns.input, f'{i:d}')
        use_i = i + 500 if fractions_check else i
        if ns.old_fs != 0:
            use_i += ns.old_fs - 500 
        f = fractions[use_i]
        if ns.inbox:
            output_complete = os.path.join(ns.output, f'inbox', 'fiducial', f'{use_i:d}', f'groups_00{ns.snapnum:d}')
        else:
            output_complete = os.path.join(ns.output, f'outbox', 'fiducial', f'{use_i:d}', f'groups_00{ns.snapnum:d}')
        
        if not os.path.exists(output_complete):
            os.makedirs(output_complete)

        FoF = read_halo(snapdir, ns.snapnum)
        
        if not ns.inbox:
            FoF_iz = read_halo(snapdir, ns.snapnum_dz)
            
        seed = use_i * ns.seed_m + ns.seed_q #to control the displacement realisations
        if ns.inbox:
            FoF_c = displace_inbox(FoF, f, dz=ns.dz, BoxSize=ns.boxsize, seed=seed)
        else:
            FoF_c = displace_outbox(FoF, FoF_iz, f, separation=ns.separation, thetax=ns.thetax, thetay=ns.thetay, thetaz=ns.thetaz, dz=ns.dz, BoxSize=ns.boxsize, seed=seed)

        FoF_name = f'group_tab_00{ns.snapnum}.0'

        readfof.writeFoFCatalog(FoF_c, os.path.join(output_complete, FoF_name))

if __name__ == '__main__':
    desc = 'Produce and save the contaminated catalogues in the fixed cosmology case'

    parser = argparse.ArgumentParser(description=desc)

    # required arguments
    group = parser.add_argument_group('required arguments')    

    h = 'the dispalcement along z'
    group.add_argument('--dz', type=float, help=h, required=True)

    h = 'file containing the interloper fraction'
    group.add_argument('--f', type=str, help=h, required=True)

    # default argument
    h = 'separation of the boxes in Mpc/h'
    parser.add_argument('--separation', type=float, default=0, help=h)

    h = 'folder containing the boxes'
    parser.add_argument('--input', type=str, default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_fiducial/FoF/', help=h)
    
    h = 'folder to save the contaminated FoF'
    parser.add_argument('--output', default='/mustfs/LAPP-DATA/theorie/cagliari/Quijote_fiducial/FoF_contaminated/', help=h)

    h = 'number of catalogues'
    parser.add_argument('--number', type=int, default=500, help=h)

    h = 'first catalogue to analyse'
    parser.add_argument('--first', type=int, default=0, help=h)

    h = 'rotation angle around x-axis [deg]'
    parser.add_argument('--thetax', type=float, default=90, help=h)

    h = 'rotation angle around y-axis [deg]'
    parser.add_argument('--thetay', type=float, default=90, help=h)
    
    h = 'rotation angle around z-axis [deg]'
    parser.add_argument('--thetaz', type=float, default=0, help=h)

    h = 'snapshot number'
    parser.add_argument('--snapnum', type=int, default=2, choices=[0,1,2,3,4], help=h)

    h = 'snapshot number from which halos are taken'
    parser.add_argument('--snapnum-dz', type=int, default=1, choices=[0,1,2,3,4], help=h)

    h = 'physical size of the box in Mpc/h'
    parser.add_argument('--boxsize', type=float, default=1000.0, help=h)

    h = 'slope of the seed line'
    parser.add_argument('--seed-m', type=int, default=2, help=h)

    h = 'intercept of the seed line'
    parser.add_argument('--seed-q', type=int, default=7, help=h)

    h = 'flag inbox displace'
    parser.add_argument('--inbox', action='store_true', help=h)
    
    h = 'flag last 500 fractions'
    parser.add_argument('--last-fs', action='store_true', help=h)

    h = 'number of catalogues already computed'
    parser.add_argument('--old-fs', type=int, default=0, help=h)

    # and go!
    main(parser.parse_args())
 
import gizmo_analysis as gizmo
import numpy as np

Rsun = 8.2
dR = 0.5
dZ = 1.0
maxT = 100

def read_snap(gal):
    # takes in galaxy (string = m12i, m12f, m12m, etc.)
    # reads in and sets fiducial coordinates
    # returns snap
    gal_info = 'fiducial_coord/' + gal + '_res7100_center.txt'
    sim_directory = '/mnt/ceph/users/firesims/fire2/metaldiff/'+gal+'_res7100'
    snap = gizmo.io.Read.read_snapshots(['gas'], 'index', 600,
                                        properties=['position',
                                                    'smooth.length', 'temperature'],
                                        assign_center=False,
                                        simulation_directory=sim_directory)
    ref = np.genfromtxt(gal_info, comments='#', delimiter=',')
    snap.center_position = ref[0]
    snap.center_velocity = ref[1]
    snap.principal_axes_vectors = ref[2:]
    for k in snap.keys():
        for attr in ['center_position', 'center_velocity', 
                     'principal_axes_vectors']:
            setattr(snap[k], attr, getattr(snap, attr))

    return snap

for gal in ['m12i', 'm12f', 'm12m']:
    

    snap = read_snap(gal)
    pos_cyl = snap['gas'].prop('host.distance.principal.cylindrical')
    rbool = np.logical_and( pos_cyl[:,0] < Rsun + dR, pos_cyl[:,0] > Rsun - dR )
    zbool = np.abs(pos_cyl[:,1]) < dZ
    Tbool = snap['gas']['temperature'] < maxT

    keys = np.where(np.logical_and(np.logical_and(rbool, zbool), Tbool))[0]
    print(gal, 2.8 * np.median( snap['gas']['smooth.length'][keys] ) /1000, ' kpc' )


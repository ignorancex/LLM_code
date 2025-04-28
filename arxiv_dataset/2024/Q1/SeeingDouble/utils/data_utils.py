# Standard Imports
import numpy as np
from matplotlib import pyplot as plt

# Energyflow package for CMS Open Data loader
import energyflow as ef
from energyflow.utils import remap_pids



def load_data(cache_dir, pt_lower, pt_upper, eta, quality, pad, x_dim = 3, momentum_scale = 250, n = 100000, amount = 1, max_particle_select = None, frac = 1.0, return_pfcs = True):

    # Load data
    specs = [f'{pt_lower} <= jet_pts <= {pt_upper}', f'abs_jet_eta < {eta}', f'quality >= {quality}']
    sim = ef.mod.load(*specs, cache_dir = cache_dir, dataset='sim', amount= amount)

    # Gen_pt for Z
    Z1 = sim.jets_f[:,sim.gen_jet_pt]
    Z = np.zeros((Z1.shape[0], 3), dtype = np.float32 )
    Z[:,0] = Z1 / momentum_scale
    Z[:,1] = sim.jets_f[:,sim.gen_jet_eta]
    Z[:,2] = sim.jets_f[:,sim.gen_jet_phi]


    # Sim_pt for X
    X = np.zeros((Z1.shape[0],3), dtype = np.float32)
    X[:,0] = sim.jets_f[:,sim.jet_pt] / momentum_scale
    X[:,1] = sim.jets_f[:,sim.jet_eta]
    X[:,2] = sim.jets_f[:,sim.jet_phi]

    # Weights
    W = sim.jets_f[:,sim.weight]

    # Event ID's
    event_ids = sim.jets_i[:,sim.evn]


    # CMS JEC's
    C = sim.jets_f[:,sim.jec]

    # PFC's
    pfcs = sim.particles


 



    # Shuffle and trim
    shuffle_indices = np.random.choice(np.arange(pfcs.shape[0]), size = int(pfcs.shape[0] * frac), replace=False)
    # pfcs = pfcs[shuffle_indices]
    # Z = Z[shuffle_indices]
    # X = X[shuffle_indices]
    # C = C[shuffle_indices]
    # event_ids = event_ids[shuffle_indices]

    pfcs = pfcs[:n]
    Z = Z[:n]
    X = X[:n]
    C = C[:n]
    W = W[:n]
    event_ids = event_ids[:n]


    # Sort indices to find pairs
    sorted = np.sort(event_ids)
    sorted_indices = event_ids.argsort()

    counter = 0
    pairs = []
    N = len(sorted)
    for (i,id) in enumerate(sorted):
        for (j, id2) in enumerate(sorted[(i+1):]):

            if id == id2:
                counter += 1
                pairs.append((i, i+1+j))
                break

            if id2 > id:
                break

    # Z = Z[sorted_indices]
    # X = X[sorted_indices]
    # C = C[sorted_indices]
    # event_ids = event_ids[sorted_indices]

    # PFC's
    dataset = np.zeros( (pfcs.shape[0], pad, x_dim), dtype = np.float32 )
    particle_counts = []
    if return_pfcs:
        for (i, jet) in enumerate(pfcs):
            size = min(jet.shape[0], pad)
            indices = (-jet[:,0]).argsort()
            dataset[i, :size, 0] = jet[indices[:size],0] / momentum_scale
            dataset[i, :size, 1] = jet[indices[:size],1]
            dataset[i, :size, 2] = jet[indices[:size],2]
            if x_dim == 4:
                dataset[i, :size, 3] = jet[indices[:size],4] # PID
            particle_counts.append(jet.shape[0])
        if x_dim == 4:
            remap_pids(dataset, pid_i = 3, error_on_unknown = False)

        for x in dataset:
            mask = x[:,0] > 0
            yphi_avg = np.average(x[mask,1:3], weights = x[mask,0], axis = 0)
            x[mask,1:3] -= yphi_avg  

    particle_counts = np.array(particle_counts)

    # Trim and shuffle
    if max_particle_select is not None:
        dataset = dataset[particle_counts < max_particle_select]
        Z = Z[particle_counts < max_particle_select]
        X = X[particle_counts < max_particle_select]
        W = W[particle_counts < max_particle_select]
        C = C[particle_counts < max_particle_select]
        event_ids = event_ids[particle_counts < max_particle_select]
        particle_counts = particle_counts[particle_counts < max_particle_select]

    shuffle_indices = np.random.choice(np.arange(dataset.shape[0]), size = int(dataset.shape[0] * frac), replace=False)


    print("X: ", X.shape, X.dtype)
    print("Y: ", Z.shape, Z.dtype)
    print("PFCs: ", dataset.shape, dataset.dtype)

    if not return_pfcs:
        return X, Z, C, particle_counts, W, event_ids, np.array(pairs)
   
    print("Max # of particles: %d" % max(particle_counts))
    return X, dataset, Z, C, particle_counts, W, event_ids, np.array(pairs)
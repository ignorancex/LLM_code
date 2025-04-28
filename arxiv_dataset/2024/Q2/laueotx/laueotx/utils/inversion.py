import numpy as np
from tqdm.auto import trange
from laueotx.utils import config as utils_config
from laueotx.detector import Detector
from laueotx.beamline import Beamline
from laueotx.grain import Grain
from laueotx import rodrigues_space, laue_math

n_dim = 3
sample_size = 4 
rot_lim = 1

def get_Llab(gr, det_tilt, det_pos, grain_rot, grain_pos, rotation_type='rp'):
    gr.beamline.detectors.rotate_and_shift(det_tilt, det_pos)
    gr.set_orientation(grain_rot)
    gr.set_position(grain_pos)
    Llab, select_bragg = gr.get_laue_diffraction_rays_lab()
    spot_pos, in_detector = gr.beamline.detectors.get_spots_positions(Llab, x0=gr.x_Omega, select_bragg=select_bragg)

    v = gr.v
    
    if rotation_type == 'rp':
        from laueotx.laue_math import r_to_U
        U = r_to_U(gr.r)
    
    elif rotation_type == 'mrp':

        from scipy.spatial.transform import Rotation
        U = Rotation.from_mrp(gr.r).as_matrix()

    w = np.dot(U, v)[0].T 


    return Llab, spot_pos, in_detector, select_bragg, w, U

def to_unit(x):

    if x.ndim==1:
        xu = x/np.linalg.norm(x)
    elif x.ndim==2:
        xu = x/np.linalg.norm(x, axis=1, keepdims=True)

    return xu

def get_sample_with_grains(n_grains, verb=False, material='FeNiMn', file_hkl='FeNiMnHKL_b.txt', sample_size=4, seed=1023, grain_pos=None, grain_rot=None, rotation_type='rp', omega_step=1):

    omegas = utils_config.get_angles({'start':0, 'end': 361, 'step': omega_step})
    
    df = Detector(detector_type='forward',
                  side_length=400, 
                  position=160, 
                  tilt=[0,0,0]) 
    
    db = Detector(detector_type='backward',
                  side_length=400, 
                  position=-160, 
                  tilt=[0,0,0]) 

    bl = Beamline(omegas=omegas, 
                  detectors=[db, df],
                  lambda_lim=[0.6, 6])

    gr = Grain(material=material, 
               file_hkl=file_hkl, 
               beamline=bl, 
               rotation_type=rotation_type)

    # import chaospy
    n_dim = 3


    np.random.seed(seed)

    if grain_pos is None:
        # from scipy.stats.qmc import LatinHypercube, scale
        # positions of the grains inside the sample
        # uniforms = [chaospy.Uniform(-sample_size, sample_size) for _ in range(n_dim)]
        # grain_pos = chaospy.J(*uniforms).sample(n_grains, rule='latin_hypercube').T
        # uniforms = LatinHypercube(d=n_dim).random(n_grains)
        # grain_pos = scale(uniforms, -sample_size, sample_size)
        grain_pos = np.random.uniform(size=(n_grains, n_dim), low=-sample_size, high=sample_size)

    if grain_rot is None:
        # rotations of the grains
        # uniforms = [chaospy.Uniform(-rot_lim, rot_lim) for _ in range(n_dim)]
        # grain_rot = chaospy.J(*uniforms).sample(n_grains, rule='latin_hypercube').T
        # uniforms = LatinHypercube(d=n_dim).random(n_grains)
        # grain_rot = scale(uniforms, -rot_lim, rot_lim)
        grain_rot = np.random.uniform(size=(n_grains, n_dim), low=-rot_lim, high=rot_lim)
        # grain_rot = np.zeros((n_grains, n_dim))
        # print('warning setting zero rotation for grains')

    # position of the detector
    det_pos = [ np.array([-160., 0., 0.]),  
                np.array([160., 0., 0.]) ]

    # tilt of the detector
    det_tilt  = [ np.array([0, 0, 0]), 
                  np.array([0, 0, 0]) ]


    list_spots_p = []
    list_plane_u = []
    list_spots_p_select = []
    list_plane_u_select = []
    list_id_ang_select = []
    list_id_hkl_select = []
    list_id_det_select = []
    list_id_grn_select = []
    list_mo_hkl_select = []


    for i in range(n_grains):

        Llab, spots_p, in_detector, select_bragg, plane_u, grain_U = get_Llab(gr, det_tilt, det_pos, grain_rot[i], grain_pos[i], rotation_type=rotation_type)

        # drop the first dimension
        spots_p = spots_p[0]
        in_detector = in_detector[0]
        select_bragg = select_bragg[0]

        # make id variables for angle, miller, detector (361, 228, 2, 3)
        n_ang, n_hkl, n_det, n_dim = spots_p.shape

        id_ang = np.tile(np.arange(n_ang).reshape(n_ang, 1, 1, 1), reps=(1, n_hkl, n_det, 1))
        id_det = np.tile(np.arange(n_det).reshape(1, 1, n_det, 1), reps=(n_ang, n_hkl, 1, 1))
        id_hkl = np.tile(np.arange(i*n_hkl, (i+1)*n_hkl).reshape(1, n_hkl, 1, 1), reps=(n_ang, 1, n_det, 1))
        ih = np.tile(np.arange(0, n_hkl).reshape(1, n_hkl, 1, 1), reps=(n_ang, 1, n_det, 1))
        # mo_hkl = gr.hkl['M'][ih]
        d = np.linalg.norm(gr.v, axis=0)
        mo_hkl = d[ih]

        select = in_detector & np.expand_dims(select_bragg, axis=-1)
        # select = select.numpy()
        # select[:] = True
        # print('WARNING: using all spots (no detector, no bragg!!)')

        spots_p_select = spots_p[select]
        id_ang_select = id_ang[select]
        id_hkl_select = id_hkl[select]
        id_det_select = id_det[select]
        mo_hkl_select = mo_hkl[select]
        id_grn_select = np.full(len(spots_p_select), i)
        plane_u_select = plane_u[id_hkl_select[:,0]-(i+1)*n_hkl,:]
            
        # no selection, all spots
        list_spots_p.append(spots_p)
        list_plane_u.append(plane_u)

        # store selected spots
        list_spots_p_select.append(spots_p_select)
        list_plane_u_select.append(plane_u_select)
        list_id_ang_select.append(id_ang_select)
        list_id_hkl_select.append(id_hkl_select)
        list_id_det_select.append(id_det_select)
        list_id_grn_select.append(id_grn_select)
        list_mo_hkl_select.append(mo_hkl_select)

    spots_p_all = np.array(list_spots_p)
    plane_u_all = np.array(list_plane_u)
    spots_p_select = np.concatenate(list_spots_p_select)
    plane_u_select = np.concatenate(list_plane_u_select)
    id_ang_select = np.concatenate(list_id_ang_select)
    id_hkl_select = np.concatenate(list_id_hkl_select)
    id_det_select = np.concatenate(list_id_det_select)
    id_grn_select = np.concatenate(list_id_grn_select)
    mo_hkl_select = np.concatenate(list_mo_hkl_select)

    if verb:
        print(f'all grains shape={spots_p_all.shape}')
        print(f'n_spots detected {len(spots_p_select)}')
        print(f'n_hkl planes detected {len(np.unique(id_hkl_select))}')


    return spots_p_all, plane_u_all, spots_p_select, id_ang_select, id_hkl_select, id_det_select, id_grn_select, mo_hkl_select, plane_u_select, grain_pos, grain_rot, gr.beamline.O, gr.v


def get_u(p, e1, Gamma):
    
    M = np.linalg.multi_dot([Gamma.T, p, e1.T, Gamma])
    Mt = (M+M.T)/2
    Lam, Q = np.linalg.eig(Mt)
    sorting = np.argsort(Lam)
    Lam = Lam[sorting]
    Q = Q[:,sorting]
    A_est = np.atleast_2d(Q[:,0]).T
    return A_est

def get_u_for_rotations(Gammas, det_spots, id_ang, id_det):

    e1 = np.array([[1, -1], [0, 0], [0,0]])

    list_u = []
    for i in range(len(det_spots)):
        p = det_spots[[i]].T
        Gamma = Gammas[id_ang[i]][0]
        e1_det = e1[:,[id_det[i,0]]]
        u = get_u(p, e1_det, Gamma)
        list_u.append(u)
    arr_u = np.array(list_u).squeeze()

    # remove symmetry
    # arr_u = arr_u * np.sign(arr_u[:,0])[:,np.newaxis]

    return arr_u


def get_u_for_shifts_opt(Gammas, p_spots, id_ang):


    from scipy.optimize import minimize
    from scipy.special import huber
    

    def obj(x0):

        u = get_u_for_x(Gammas, p_spots, x0, id_ang)

        # list_u = []
        # for i in range(len(p_spots)):
        #     Gamma = Gammas[id_ang[i]][0]
        #     p = p_spots[[i]].T - np.expand_dims(np.dot(Gamma, x0), axis=-1)
        #     u = get_u(p, e1, Gamma)
        #     list_u.append(u)
        # u = np.array(list_u).squeeze()
        # u *= np.sign(u[:,[0]])                               
        
        um = np.mean(u, axis=0)

        # x_center = np.array([0,0,0])
        # u0 = get_u_for_x(Gammas, p_spots, x_center, id_ang, e1)
        # um0 = np.mean(u0, axis=0)



        cost = np.mean((u-um)**2)
        # cost0 = np.mean((u-um0)**2)
        # cost = np.mean(huber(1e-4, u-um))
        return cost

    x0 = np.array([0,0,0])
    
    res = minimize(obj, x0=x0, method='CG', tol=1e-12, options=dict(maxiter=100))
    x_best = res.x
    # print('---')
    # print('x_best', x_best, '{: 2.4e}'.format(obj(x_best)))
    # print('x0    ', x0, '{: 2.4e}'.format(obj(x0)))
    cost_best = obj(x_best)
    cost_init = obj(np.array([0,0,0]))

    # u_best = get_u_for_x(Gammas, p_spots, x_best, id_ang, e1)
    # u_init = get_u_for_x(Gammas, p_spots, x0, id_ang, e1)
    # umb = np.mean(u_best, axis=0)
    # umi = np.mean(u_init, axis=0)
    # print(np.mean((umb-umi))**2)

    return x_best, cost_best, cost_init

def get_u_for_x(Gammas, p_spots, x0, id_ang):

    e1 = np.expand_dims(np.array([1, 0, 0]), axis=-1) # incoming beam aligned with x axis

    list_u = []
    for i in range(len(p_spots)):
        Gamma = Gammas[id_ang[i]][0]
        # p = p_spots[[i]].T - np.expand_dims(np.dot(Gamma, x0), axis=-1)
        p = p_spots[[i]].T - np.dot(Gamma, x0)
        u = get_u(p, e1, Gamma)
        list_u.append(u)
    u = np.array(list_u).squeeze()
    u *= np.sign(u[:,[0]])                               
    return u

def get_u_mean(p, x, Gammas, id_ang, u_sign=-1):
        
    e1 = np.expand_dims(np.array([1, 0, 0]), axis=-1) # incoming beam aligned with x axis

    list_M = []
    for i in range(len(p)):
        Gamma = Gammas[id_ang[i]][0]
        p_bar = p[[i]].T - np.dot(Gamma, x)
        M = np.linalg.multi_dot([Gamma.T, p_bar, e1.T, Gamma])
        list_M.append(M)
    M = np.sum(list_M, axis=0)
    Mt = (M+M.T)/2
    Lam, Q = np.linalg.eigh(Mt)
    sorting = np.argsort(Lam)
    Lam = Lam[sorting]
    Q = Q[:,sorting]
    u = np.atleast_2d(Q[:,0]).T
    u *= np.sign(u[0])*u_sign
    return u    
    

def get_x_for_multiple_hkl_planes(u_hkl_est, label_spot_hkl, Gammas, p_spots, id_ang, test=False):

    n_labels = len(u_hkl_est)
    # n_labels = 10
    n_dim = 3
    x_est = np.zeros((n_labels, n_dim))
    cost_best = np.zeros(n_labels)
    cost_init = np.zeros(n_labels)
    u_hkl_best = np.zeros_like(u_hkl_est)
    
    if test:
        n_labels = 1
        print('test!')

    for i in trange(n_labels):
        select = label_spot_hkl==i
        x_est[i], cost_best[i], cost_init[i] = get_u_for_shifts_opt(Gammas, p_spots[select], id_ang[select])
        ux = get_u_for_x(Gammas, p_spots[select], x_est[i], id_ang[select])
        u_hkl_best[i] = np.mean(ux, axis=0) 
    return x_est, u_hkl_best, cost_best, cost_init


def get_ux_for_multiple_hkl_planes(u_hkl_est, label_spot_hkl, Gammas, p_spots, id_ang, test=False, x0=np.array([[0],[0],[0]])):

    n_labels = len(u_hkl_est)
    # n_labels = 10
    n_dim = 3
    x_est = np.zeros((n_labels, n_dim))
    u_est = np.zeros((n_labels, n_dim))
    cost_best = np.zeros(n_labels)
    cost_init = np.zeros(n_labels)
    u_hkl_best = np.zeros_like(u_hkl_est)
    
    if test:
        n_labels = 1
        print('test!')

    # for i in trange(n_labels):
    for i in trange(1):

        # get the hkl plane
        select = label_spot_hkl==i

        # run elbo iterationss
        u_est[i], x_est[i], cost_best[i], cost_init[i] = get_ux_elbo(Gammas, p_spots[select], id_ang[select], x0)

        
    return x_est, u_est, cost_best, cost_init




def get_ux_elbo_old(Gammas, p_spots, id_ang, x0 = np.array([[0],[0],[0]])):


    def get_u_for_x(Gammas, p_spots, x, id_ang):

        e1 = np.expand_dims(np.array([1, 0, 0]), axis=-1) # incoming beam aligned with x axis

        list_u = []
        for i in range(len(p_spots)):
            Gamma = Gammas[id_ang[i]][0]
            p = p_spots[[i]].T - np.dot(Gamma, x)
            u = get_u(p, e1, Gamma)
            list_u.append(u)
        u = np.array(list_u)
        u *= np.sign(u[:,[0],:])
        u = np.mean(u, axis=0)
        return u

    def get_x_for_u(Gammas, ps, u, x, id_ang):

        e1 = np.expand_dims(np.array([1, 0, 0]), axis=-1) # incoming beam aligned with x axis

        n_dim = 3
        I = np.eye(n_dim)
        list_dx = []
        # ps_unit = to_unit(ps)
        # print('ps_unit.shape', ps_unit.shape)
        for i in range(len(ps)):
        # for i in range(1):
            print('============', i)

            Gamma = Gammas[id_ang[i]][0]
            # Lt = I - 2*np.linalg.multi_dot([Gamma, u, u.T, Gamma.T])

            v = np.dot(Gamma, u)
            Lt = householder_transform(v)
            
            p_data = ps[[i]].T 
            # print('p_data', p_data.shape)
            p_data_norm = np.linalg.norm(p_data+x)

            # p_data = p_data/p_data_norm
            # p = np.dot(Gamma, x) + p_data_norm*np.dot(Lt, e1)
        
            p = np.dot(Lt, e1)
            p = p/np.linalg.norm(p) 
            p = p*p_data_norm + np.dot(Gamma, x)
            # print('Gamma ', Gamma.shape, Gamma)
            dx = p_data - p
            
            print('u', u.shape, u.T)
            print('x', x.shape, x.T)
            print('p_data', p_data.shape, p_data.T)
            print('p_est', p.shape, p.T)
            print('p_data_norm', p_data_norm.shape, p_data_norm.T)
            print('p', p.shape, p.T)
            print('dx', dx.shape, dx.T)
            
            list_dx.append(dx)


        dx = np.mean(np.array(list_dx), axis=0)
        print('dx', dx.shape, dx)
        return dx

    def householder_transform(v):
        v = np.atleast_2d(v)
        if v.shape[1] != 1:
            v = v.T
        v = v/np.linalg.norm(v)
        return np.eye(v.shape[0]) - 2*np.dot(v, v.T)
        

    print('x0', x0.shape, x0)

    n_iter = 10
    x = x0
    for i in range(n_iter):
        print(f'==== iter {i}')
        u = get_u_for_x(Gammas, p_spots, x, id_ang)
        print('u', u.shape, u.T)
        x = get_x_for_u(Gammas, p_spots, u, x, id_ang)
        print('x', x.shape, x.T)

    return u, x, None, None





def scatter_spots(list_spot_positions, same_ax=False, lims=[-1, 1], projections=True, markers=None, figsize=10, title='', **kwargs):

    import pylab as plt
    
    if markers is None:
        markers = ['o']*len(list_spot_positions)
        
    if type(list_spot_positions) is not list:
        list_spot_positions = [list_spot_positions]

    def add_scatter_plot(ax, spot_positions, **kwargs): 

        kwargs.setdefault('c', np.arange(len(spot_positions)))

        ax.scatter(spot_positions[:,0], spot_positions[:,1], spot_positions[:,2], cmap='seismic', **kwargs)
        ax.set(xlim=lims, ylim=lims, zlim=lims)
        if projections:
            for i in range(3):
                spot_positions_proj = spot_positions.copy() 
                if i==1:
                    spot_positions_proj[:,i] = 1
                else:
                    spot_positions_proj[:,i] = -1
                ax.scatter(spot_positions_proj[:,0], spot_positions_proj[:,1], spot_positions_proj[:,2], c=np.arange(len(spot_positions_proj)), cmap='gray')
        ax.scatter(xs=np.linspace(0,1,100), ys=[0], zs=[0], c='k', marker='.', zorder=0)
        ax.scatter(xs=[1], ys=[0], zs=[0], c='k', marker='>', s=100, zorder=0)
        ax.scatter(xs=[0], ys=[0], zs=[0], c='k', marker='o', s=100, zorder=0)
        ax.set(xlabel='x', ylabel='y', zlabel='z', title=title)
       
    
    nx, ny = 1, 1 if same_ax else len(list_spot_positions); fig, ax = plt.subplots(nx, ny, figsize=(ny * figsize, nx * figsize), squeeze=False, subplot_kw={'projection':'3d'}); axc=ax[0,0]; axl=ax[0,:];
    for i, s in enumerate(list_spot_positions):
        ax_use = axl[0] if same_ax else axl[i]
        add_scatter_plot(ax_use, s, marker=markers[i], **kwargs)

def hist_2d_flat(arr_u, bins=100, **kw):

    import pylab as plt

    nx, ny = 1, 3; fig, ax = plt.subplots(nx, ny, figsize=(ny * 8, nx * 6), squeeze=False); axc=ax[0,0]; axl=ax[0,:];
    pairs = [[0,1], [0,2], [1,2]]
    for i in range(3):
        h = axl[i].hist2d(arr_u[:,pairs[i][0]], arr_u[:,pairs[i][1]], cmap='Spectral_r', bins=bins, **kw)    
        fig.colorbar(h[3], ax=axl[i])


def scatter_projections(x, ax=None, figsize=[8,6], **kw):

    import pylab as plt

    if ax is None:
        nx, ny = 1, 3; fig, ax = plt.subplots(nx, ny, figsize=(ny * figsize[0], nx * figsize[1]), squeeze=False); axc=ax[0,0]; axl=ax[0,:];
    else:
        axl = ax.ravel()

    pairs = [[0,1], [0,2], [1,2]]
    for i in range(3):
        h = axl[i].scatter(x[:,pairs[i][0]], x[:,pairs[i][1]], cmap='Spectral_r', **kw)    

    return ax

def db_scan_clustering(X, labels_true=None, eps=0.05, min_samples=5, verb=False):
    
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from sklearn import metrics

    scaler = StandardScaler().fit(X)
    X_tr = scaler.transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_tr)

    # db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    n_labels = len(np.unique(labels))

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    if labels_true is not None:
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
    
    unique_labels = np.sort(np.unique(labels))
    for i, lab in enumerate(unique_labels):
        select = labels == lab
        if verb:
            print(f'label {lab} n={np.count_nonzero(select)}')

    if unique_labels[0] == -1:
        unique_labels = unique_labels[1:]
    
    X_center = np.zeros((len(unique_labels),3))
    for i, lab in enumerate(unique_labels):
        
        select = labels == lab
        X_center[lab] = np.median(X[select,:], axis=0)

    X_center = scaler.inverse_transform(X_center)
    
    return X_center, labels


def assign_hkl_est_to_tru(u_plane, u_hkl_est, label_spot_hkl, id_hkl, id_grn):


    n_hkl = len(np.unique(id_hkl))
    n_dim = 3

    u_hkl_est_label = fix_symmetry(u_hkl_est[label_spot_hkl])
    # print(u_hkl_est_label.shape)

    # for i in range(len(u_plane)):
    u_plane_est = np.zeros((n_hkl, n_dim))
    u_plane_tru = np.zeros((n_hkl, n_dim))
    # print(u_plane_est.shape, u_plane.shape)

    u_plane_tru = fix_symmetry(to_unit(u_plane))
    for i in range(10):

        select = id_hkl[:,0] == i
        # print(np.count_nonzero(select), select.shape, u_hkl_est_label[select].shape)

        ue = fix_symmetry(to_unit(u_hkl_est_label[select]))
        ut = fix_symmetry(to_unit(u_plane[select]))
        # print(ue)
        # print(ut)

        u_plane_est[i,:] = np.median(ue, axis=0)
        u_plane_tru[i,:] = np.median(ut, axis=0)
        # print(i, np.count_nonzero(select), to_unit(u_plane_est[i,:]), u_plane_tru[i,:])

    return u_plane_est, u_plane_tru


def fix_symmetry(x):

    if x.ndim==1:
        x *= np.sign(x[0])
    elif x.ndim==2:
        x *= np.sign(x[:,0, np.newaxis])

    return x


def assignment(x, y):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import pairwise_distances
    cost = pairwise_distances(x, y)
    row_ind, col_ind = linear_sum_assignment(cost)
    x_match, y_match = x[row_ind], y[col_ind]
    return x_match, y_match

def plot_plane_match(w_tru, w_est, title=''):

    import pylab as plt
    bins_x = np.linspace(0, 1, 100)
    bins_y = np.linspace(-1e-1, 1e-1, 100)

    nx, ny = 1, 3; fig, ax = plt.subplots(nx, ny, figsize=(ny * 8, nx * 6), squeeze=False); axc=ax[0,0]; axl=ax[0,:];
    for i in range(3):
        # axl[i].hist2d(w_tru[:,i], w_est[:,i]-w_tru[:,i], bins=[bins_x, bins_y])
        axl[i].scatter(w_tru[:,i], w_est[:,i]-w_tru[:,i])
        axl[i].axhline(0, c='k')
        axl[i].grid(True)
    fig.suptitle(title)

def hist_plane_match(u_plane_diff, log=False):

    import pylab as plt

    n_dim = 3
    bins = np.linspace(-5,5,100)
    nx, ny = 1, n_dim; fig, ax = plt.subplots(nx, ny, figsize=(ny * 8, nx * 6), squeeze=False); axc=ax[0,0]; axl=ax[0,:];
    for i in range(n_dim):
        axl[i].hist(u_plane_diff[:,i], bins=bins)
        axl[i].set(xlabel=f'u[{i}] estimated - true', ylabel='number of hkl planes (one per spot)')
        if log:
            axl[i].set(yscale='log')
        axl[i].grid(True)



def wahba_svd(w, v, a):
    """
    https://en.wikipedia.org/wiki/Wahba's_problem
    J(\mathbf {R} )={\frac {1}{2}}\sum _{k=1}^{N}a_{k}\|\mathbf {w} _{k}-\mathbf {R} \mathbf {v} _{k}\|^{2}} for { N\geq 2}{ N\geq 2}
    """
    
    w = w*np.expand_dims(np.sqrt(a), axis=-1)
    v = v*np.expand_dims(np.sqrt(a), axis=-1)
    M = np.dot(v.T, w)
    U, S, Vh = np.linalg.svd(M)

    detU = np.linalg.det(U)
    detVh = np.linalg.det(Vh)

    H = np.diag([1, 1, detU*detVh])
    R = np.dot(U, np.dot(H, Vh))
    # R = np.dot(U, Vh)
    
    return R

def get_v_ref(unique_hkl=True, unit=True):

    omegas = utils_config.get_angles({'start':0, 'end': 361, 'step': 1})

    df = Detector(detector_type='forward',
                  side_length=400, 
                  position=160, 
                  tilt=[0,0,0]) 
    
    db = Detector(detector_type='backward',
                  side_length=400, 
                  position=-160, 
                  tilt=[0,0,0]) 

    bl = Beamline(omegas=omegas, 
                  detectors=[db, df],
                  lambda_lim=[0.6, 6])

    gr = Grain(material='FeNiMn', file_hkl='FeNiMnHKL_b.txt', beamline=bl)
        
    v_ref = gr.v.T
    
    if unit:
        v_ref = to_unit(v_ref)

    if unique_hkl:        
        _, ind = np.unique(fix_symmetry(v_ref), axis=0, return_index=True)
        v_ref = v_ref[ind]

    return v_ref



def get_grain_rotation_wahba(u_est, n_div=10, wahba=True, verb=False, assignment='linear_sum', unique_hkl=True):
    
    
    from laueotx.rodrigues_space import divide_rodrigues_space
    from sklearn.metrics import pairwise_distances
    from laueotx.laue_math import r_to_U

    def assign(u, u_est):

        if assignment == 'linear_sum':
            
            from sklearn.metrics import pairwise_distances
            from scipy.optimize import linear_sum_assignment
            C = pairwise_distances(u, u_est, metric='cosine')
            # C[C>1]=1
            ir, ic = linear_sum_assignment(C)

        return ir, ic


    v_ref = get_v_ref(unique_hkl)
    r = divide_rodrigues_space(n_div)
    U = r_to_U(r)
    Uc = U.copy()
    # v_ref = fix_symmetry(v_ref)
    # u_est = fix_symmetry(u_est)
    
    cost = np.zeros(len(r))

    range_fun = trange if verb else range

    for i in range_fun(len(U)):
        

        u = np.dot(U[i], v_ref.T).T
        # u = fix_symmetry(u)
        ir, ic = assign(u, u_est)
        u1 = u[ir]
        u2 = u_est[ic]
        cost[i] = np.sum((u1 - u2)**2)
        
        if wahba:

            # refine with inverse
            R = wahba_svd(u1, u2, a=np.ones(len(u1)))
            Uc[i] = np.dot(R, U[i])
            u = np.dot(Uc[i], v_ref.T).T
            # u = fix_symmetry(u)
            ir, ic = assign(u, u_est)
            u1 = u[ir]
            u2 = u_est[ic]
            cost[i] = np.sum((u1 - u2)**2)
    
    id_best = np.argmin(cost)
    U_best = Uc[id_best]
    r_best = r[id_best]

    # u = np.dot(U_best, v_ref.T).T
    # C = pairwise_distances(u, u_est, metric='cosine')
    # ir, ic = linear_sum_assignment(C)
    # u1 = u[ir]
    # u2 = u_est[ic]
    # cc = np.sum((u1 - u2)**2)
        
    return U_best, r_best, cost




        
def assemble_hkl_into_grain(u_hkl, label_grain, n_div=8, wahba=True, verb=False):

    u_hkl = to_unit(u_hkl)

    n_grains_est = np.max(label_grain)+1

    list_U = []
    list_r = []
    for i in trange(n_grains_est):
        select = label_grain == i
        U, r, cost = get_grain_rotation_wahba(u_hkl[select], n_div=n_div, wahba=wahba, verb=verb)
        list_U.append(U)
        list_r.append(r)
        print('grain={} hkl assignment cost {:2.4e}'.format(i, np.min(cost)))

    U = np.array(list_U)
    r = np.array(list_r)
    return U, r


        
def get_matrix_diff_with_symmetries(Uc, U):

    # list of crystallographic symmetries
    F1 = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]] )
    F2 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]] )
    Fs = [F1, F2]
    
    diff = np.zeros(len(Fs))
    Ucfs = np.empty(len(Fs), dtype=object)
    for i, F in enumerate(Fs):
        Ucfs[i] = np.dot(Uc, F.T)
        diff[i] = np.sum((Ucfs[i]-U)**2)
    id_best = np.argmin(diff)
    return Ucfs[id_best]
    
def plot_projections_hist(x, **kw):
    import pylab as plt

    n_dim = x.shape[1]
    nx, ny = 1, n_dim; fig, ax = plt.subplots(nx, ny, figsize=(ny * 16, nx * 10), squeeze=False); axc=ax[0,0]; axl=ax[0,:];

    for i in range(n_dim):

        h = axl[i].hist2d(x[:,i], x[:,(i+1)%n_dim], **kw)
        fig.colorbar(h[3], ax=axl[i])


def plot_projections_scatter(x, axl=None, figsize=[12,8], lim=None, **kw):
    import pylab as plt

    n_dim = x.shape[1]

    if axl is None:
        nx, ny = 1, n_dim; fig, ax = plt.subplots(nx, ny, figsize=(ny * figsize[0], nx * figsize[1]), squeeze=False); axc=ax[0,0]; axl=ax[0,:];

    fig = axl[0].figure

    for i in range(n_dim):

        i1 = i 
        i2 = (i+1)%n_dim
        axl[i].scatter(x[:,i1], x[:,i2], **kw)
        axl[i].set(xlabel=f'dim {i1}', ylabel=f'dim {i2}')
        axl[i].legend()
        axl[i].grid(True)

        if lim is not None:
            axl[i].set(xlim=[-lim, lim], ylim=[-lim, lim])


    return axl






def get_shifted_u(Gammas, det_spots, id_ang, label_spot_hkl, x_grain):
    # get_u_for_rotations(Gammas, det_spots, id_ang)

    n_hkl = np.max(label_spot_hkl)+1
    n_dim = det_spots.shape[1]

    assert len(x_grain) == n_hkl, f"{len(x_grain)}, {n_hkl}"

    e1 = np.expand_dims(np.array([1, 0, 0]), axis=-1) # incoming beam aligned with x axis
    u_hkl = np.zeros((n_hkl, n_dim))

    list_ux = []
    for i in range(n_hkl):


        select = label_spot_hkl==i
        x = x_grain[i]
        
        det_spots_select = det_spots[select]
        id_ang_select = id_ang[select]

        # shift for the grain position
        ux = get_u_for_x(Gammas, det_spots_select, x, id_ang, e1)
        list_ux.append(np.mean(ux, axis=0))

    return np.array(list_ux)



        

    


def align_rotation_matrices(R_src, R_dst):

    from scipy.linalg import orthogonal_procrustes
    R = orthogonal_procrustes(R_src, R_dst)[0]

    return np.dot(R_src, R)




def rotation_est_tru_match_spots_procrustes(U_est, U_tru):
    
    from scipy.linalg import orthogonal_procrustes
    from scipy.optimize import linear_sum_assignment

    matrix_diff = np.zeros((len(U_est), len(U_tru)))
    
    for i, Ue in enumerate(U_est):
        for j, Ut in enumerate(U_tru):
            R = orthogonal_procrustes(U_est[i], U_tru[j])[0]
            # print('-----')
            # print(R)
            R = np.round(R)
            R[R==0]=0
            # print(R)
            U_est_rot = np.dot(U_est[i], R)
            matrix_diff[i,j] = np.linalg.norm(U_est_rot-U_tru[j], 'fro')

    # print(matrix_diff)
    ir, ic = linear_sum_assignment(matrix_diff)
    # print(ir, ic)
    
    U_est_match = U_est[ir]
    U_tru_match = U_tru[ic]
    
    U_est_match_rot = []
    n = np.min([len(U_est_match), len(U_tru_match)])
    for i in range(n):
        R = orthogonal_procrustes(U_est_match[i], U_tru_match[i])[0]
        R = np.round(R)
        R[R==0]=0
        U_est_best = np.dot(U_est_match[i], R)
        U_est_match_rot.append(U_est_best)
        print(np.linalg.norm(U_est_match_rot[i]-U_tru_match[i], 'fro'))
    U_est_match_rot = np.array(U_est_match_rot)
    return U_est_match_rot[ir], U_tru_match[ic]
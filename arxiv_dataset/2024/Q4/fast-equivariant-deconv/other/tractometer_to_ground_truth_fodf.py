import nibabel as nib
import numpy as np
import scipy.stats as st
from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix
import numpy as np
from scipy import special as sci



def _sh_matrix(sh_degree, vector, with_order=1):
    """
    Create the matrices to transform the signal into and from the SH coefficients.
    A spherical signal S can be expressed in the SH basis:
    S(theta, phi) = SUM c_{i,j} Y_{i,j}(theta, phi)
    where theta, phi are the spherical coordinates of a point
    c_{i,j} is the spherical harmonic coefficient of the spherical harmonic Y_{i,j}
    Y_{i,j} is the spherical harmonic of order i and degree j
    We want to find the coefficients c from N known observation on the sphere:
    S = [S(theta_1, phi_1), ... , S(theta_N, phi_N)]
    For this, we use the matrix
    Y = [[Y_{0,0}(theta_1, phi_1)             , ..., Y_{0,0}(theta_N, phi_N)                ],
        ................................................................................... ,
        [Y_{sh_order,sh_order}(theta_1, phi_1), ... , Y_{sh_order,sh_order}(theta_N, phi_N)]]
    And:
    C = [c_{0,0}, ... , c_{sh_order,sh_order}}
    We can express S in the SH basis:
    S = C*Y
    Thus, if we know the signal SH coefficients C, we can find S with:
    S = C*Y --> This code creates the matrix Y
    If we known the signal Y, we can find C with:
    C = S * Y^T * (Y * Y^T)^-1  --> This code creates the matrix Y^T * (Y * Y^T)^-1
    Parameters
    ----------
    sh_degree : int
        Maximum spherical harmonic degree
    vector : np.array (N_grid x 3)
        Vertices of the grid
    with_order : int
        Compute with (1) or without order (0)
    Returns
    -------
    spatial2spectral : np.array (N_grid x N_coef)
        Matrix to go from the spatial signal to the spectral signal
    spectral2spatial : np.array (N_coef x N_grid)
        Matrix to go from the spectral signal to the spatial signal
    """
    if with_order not in [0, 1]:
        raise ValueError('with_order must be 0 or 1, got: {0}'.format(with_order))
    x, y, z = vector[:, 0], vector[:, 1], vector[:, 2]
    colats = np.arccos(z)
    lons = np.arctan2(y, x) % (2 * np.pi)
    grid = (colats, lons)
    gradients = np.array([grid[0].flatten(), grid[1].flatten()]).T
    num_gradients = gradients.shape[0]
    if with_order == 1:
        num_coefficients = int((sh_degree + 1) * (sh_degree/2 + 1))
    else:
        num_coefficients = sh_degree//2 + 1
    b = np.zeros((num_coefficients, num_gradients))
    for id_gradient in range(num_gradients):
        id_column = 0
        for id_degree in range(0, sh_degree + 1, 2):
            for id_order in range(-id_degree * with_order, id_degree * with_order + 1):
                gradients_phi, gradients_theta = gradients[id_gradient]
                y = sci.sph_harm(np.abs(id_order), id_degree, gradients_theta, gradients_phi)
                if id_order < 0:
                    b[id_column, id_gradient] = np.imag(y) * np.sqrt(2)
                elif id_order == 0:
                    b[id_column, id_gradient] = np.real(y)
                elif id_order > 0:
                    b[id_column, id_gradient] = np.real(y) * np.sqrt(2)
                id_column += 1
    b_inv = np.linalg.inv(np.matmul(b, b.transpose()))
    spatial2spectral = np.matmul(b.transpose(), b_inv)
    spectral2spatial = b
    return spatial2spectral, spectral2spatial

def get_vector_and_coord(start, end):
    if np.sum(np.abs(np.floor(end) - np.floor(start)))>0:
        sorted_coord = np.sort(np.vstack((start, end)), 0)
        sorted_coord[0] = np.ceil(sorted_coord[0])
        sorted_coord[1] = np.floor(sorted_coord[1])
        all_t = []
        for i in range(3):
            int_coord = np.arange(sorted_coord[0, i], sorted_coord[1, i]+1)
            t = (int_coord - start[i]) / ((end[i] - start[i]))
            all_t.append(t)
        all_t = np.sort(np.unique(np.concatenate(all_t)))
        all_t = all_t[(all_t>0)*(all_t<1)]
        if len(all_t)>0:
            list_vec = [start]
            for t in all_t:
                new_coord = start + t*(end - start)
                list_vec.append(new_coord)
            list_vec.append(end)
            list_vec = np.vstack(list_vec)
            voxel_coord = []
            is_increasing = ((end - start)>0)
            for i in range(3):
                if is_increasing[i]:
                    voxel_coord.append(np.floor(list_vec[:-1, i]))
                else:
                    voxel_coord.append(np.floor(list_vec[1:, i]))
            voxel_coord = np.vstack(voxel_coord).T
            list_vec = list_vec[1:] - list_vec[:-1]
            return list_vec, voxel_coord.astype(int)
    list_vec = (end - start)[None]
    voxel_coord = np.floor(start)[None].astype(int)
    return list_vec, voxel_coord.astype(int)


data_path_gt = f'/scratch/age261/ismrm/'
gt_mask = nib.load(f'{data_path_gt}/ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2/NoArtifacts_Relaxation.nii.gz')
header = gt_mask.header
affine = gt_mask.affine
gt_mask = gt_mask.get_fdata()
H, W, Z, B = gt_mask.shape


n_fiber_max = 11000
track = nib.streamlines.load(f'{data_path_gt}/FilesForSimulation/Fibers.trk')

all_fiber_raw = np.zeros((H, W, Z, n_fiber_max, 3), dtype=np.float16)
all_fiber_length = np.zeros((H, W, Z, n_fiber_max), dtype=np.float16)
n_fiber = np.zeros((H, W, Z)).astype(int)
miss = np.zeros((H, W, Z)).astype(int)
stream = track.tractogram.streamlines
for st in range(len(stream)):
    if st%1000==0:
        print(st/len(stream)*100, end='\r')
    fiber = track.tractogram.streamlines[st]# + 0.5
    fiber = (fiber - np.array([123.5, 153.5, -0.5])[None] + np.array([178, 214, 0])[None])/2
    #fiber = (fiber - np.array([124, 154, 0])[None] + np.array([178, 214, 0])[None])/2
    fiber[:, 0] = H - fiber[:, 0]
    fiber[:, 1] = W - fiber[:, 1]
    for f in range(len(fiber)-1):
        start = fiber[f]
        end = fiber[f+1]
        list_vec, voxel_coord = get_vector_and_coord(start, end)
        list_vec[:, 0] = - list_vec[:, 0]
        list_vec[:, 1] = - list_vec[:, 1]
        for sel_f in range(list_vec.shape[0]):
            i, j, k = voxel_coord[sel_f]
            vec = list_vec[sel_f]
            vec_norm = np.linalg.norm(vec)
            assert vec_norm!=0
            vec_normed = vec / vec_norm
            n = n_fiber[i, j, k]
            if n<n_fiber_max:
                all_fiber_raw[i, j, k, n] = vec
                all_fiber_length[i, j, k, n] = vec_norm
                n_fiber[i, j, k] += 1
            else:
                miss[i, j, k] += 1


print(np.max(miss))

n_max = n_fiber.max()
all_fiber_raw = all_fiber_raw[:, :, :, :n_max]
all_fiber_length = all_fiber_length[:, :, :, :n_max]

clipped_img = nib.Nifti1Image(n_fiber, affine, header)
nib.save(clipped_img, f'{data_path_gt}/FilesForSimulation/ground_truth_from_streamlines/n_fiber.nii.gz')

#clipped_img = nib.Nifti1Image(all_fiber_raw, affine, header)
#nib.save(clipped_img, f'{data_path_gt}/FilesForSimulation/ground_truth_from_streamlines/all_fiber_raw.nii.gz')
#clipped_img = nib.Nifti1Image(all_fiber_length, affine, header)
#nib.save(clipped_img, f'{data_path_gt}/FilesForSimulation/ground_truth_from_streamlines/all_fiber_length.nii.gz')



for angle_threshold in [15]:
    sh_degree = 16
    n_side = 8
    G = SphereHealpix(n_side, nest=True, k=20)
    vec = G.coords
    s2sh, sh2s = _sh_matrix(sh_degree=sh_degree, vector=vec, with_order=1)
    fODF = np.zeros((H, W, Z, s2sh.shape[1]))
    for i in range(H):
        print(i, end='\r')
        for j in range(W):
            for k in range(Z):
                n_f = n_fiber[i,j,k]
                if n_f!=0:
                    gt_fib = all_fiber_raw[i,j,k,:n_f].astype(np.float32)
                    gt_pve = np.linalg.norm(gt_fib, axis=-1)
                    #assert np.sum(gt_pve == 0)==0
                    gt_fib = gt_fib /  (gt_pve[..., None] + 1e-16)
                    fODF[i,j,k] = np.sum(((np.arccos(np.minimum(np.abs(gt_fib.dot(vec.T)), 1))*180/np.pi)<angle_threshold)*gt_pve[..., None], 0).dot(s2sh)
                    #fODF[i,j,k] = np.sum(((np.arccos(np.minimum(np.abs(gt_fib.dot(vec.T)), 1))*180/np.pi)<angle_threshold), 0).dot(s2sh)
    fODF = fODF/(np.linalg.norm(fODF, axis=-1).max() + 1e-16) / np.sqrt(3.14)
    clipped_img = nib.Nifti1Image(fODF, affine, header)
    nib.save(clipped_img, f'{data_path_gt}/FilesForSimulation/ground_truth_from_streamlines/fODF_anglethres_{angle_threshold}_shdegree_{sh_degree}_2.nii.gz')
    print(f'sh2peaks {data_path_gt}/FilesForSimulation/ground_truth_from_streamlines/fODF_anglethres_{angle_threshold}_shdegree_{sh_degree}_2.nii.gz {data_path_gt}/FilesForSimulation/ground_truth_from_streamlines/peaks_anglethres_{angle_threshold}_shdegree_{sh_degree}.nii.gz -num 10 -force')
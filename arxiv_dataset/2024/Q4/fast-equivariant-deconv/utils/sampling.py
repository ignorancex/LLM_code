import numpy as np
import math
import torch

from .spherical_harmonic import _sh_matrix
from .spherehealpix import SphereHealpix
from .laplacian import prepare_laplacian, scipy_csr_to_sparse_tensor
from .pooling import HealpixPooling, MixedPooling, SpatialPooling, IdentityPooling

class Sampling:
    """Spherical sampling class.
    """

    def __init__(self, vectors, sh_degree=None, max_sh_degree=None, constant=False, symmetric=True):
        """Initialize symmetric sampling class.
        Args:
            vectors (np.array): [V x 3] Sampling position on the unit sphere (bvecs)
            sh_degree (int, optional): Spherical harmonic degree of the sampling
            max_sh_degree (int, optional): Max Spherical harmonic degree of the sampling if sh_degree is None
            constant (bool, optional): In the case of a shell==0
        """
        # Load sampling
        assert vectors.shape[1] == 3
        self.vectors = vectors # V x 3

        # Compute sh_degree
        if sh_degree is None:
            sh_degree = 2*int((np.sqrt(8*vectors.shape[0]-7) - 3) / 4) # We want the number of SHC to be at most the number of vectors
            if not max_sh_degree is None:
                sh_degree = min(sh_degree, max_sh_degree)
        if constant:
            self.S2SH = np.ones((vectors.shape[0], 1)) * math.sqrt(4*math.pi) / vectors.shape[0] # V x 1
            self.SH2S = np.zeros(((sh_degree+1)*(sh_degree//2+1), vectors.shape[0])) # (sh_degree+1)(sh_degree//2+1) x V 
            self.SH2S[0] = 1 / math.sqrt(4*math.pi)
        else:
            # Compute SH matrices
            _, self.SH2S = self.sh_matrix(sh_degree, vectors, with_order=1) # (sh_degree+1)(sh_degree//2+1) x V 
            
            # We can't recover more SHC than the number of vertices:
            sh_degree_s2sh = 2*int((np.sqrt(8*vectors.shape[0]-7) - 3) / 4)
            sh_degree_s2sh = min(sh_degree_s2sh, sh_degree)
            if not max_sh_degree is None:
                sh_degree_s2sh = min(sh_degree_s2sh, max_sh_degree)
            self.S2SH, _ = self.sh_matrix(sh_degree_s2sh, vectors, with_order=1) # V x (sh_degree_s2sh+1)(sh_degree_s2sh//2+1)

    def sh_matrix(self, sh_degree, vectors, with_order):
        return _sh_matrix(sh_degree, vectors, with_order)



class HealpixSampling:
    """Graph Spherical sampling class.
    """
    def __init__(self, n_side, depth, patch_size, sh_degree=None, pooling_mode='average', pooling_name='mixed', hemisphere=False, legacy=False):
        """Initialize the sampling class.
        Args:
            n_side (int): Healpix resolution
            depth (int): Depth of the encoder
            sh_degree (int, optional): Spherical harmonic degree of the sampling
            pooling_mode (str, optional): specify the mode for pooling/unpooling.
                                            Can be max or average. Defaults to 'average'.
        """
        print('-'*50)
        print('-'*10, f' Create Healpix Sampling ', '-'*10)
        print(f'Healpix resolution: {n_side}')
        print(f'Depth: {depth}')
        print(f'Patch size: {patch_size}')
        print(f'Spherical harmonic degree: {sh_degree}')
        print(f'Pooling mode: {pooling_mode}')
        print(f'Pooling name: {pooling_name}')
        print(f'Hemisphere: {hemisphere}')
        print(f'Legacy: {legacy}')

        assert math.log(n_side, 2).is_integer()
        #assert n_side / (2**(depth-1)) >= 1
        if legacy:
            raise NotImplementedError('Legacy is not implemented')
            assert not hemisphere
        G = SphereHealpix(n_side, nest=True, k=8) # Highest resolution sampling
        if hemisphere:
            eps = 1e-10
            index_north_hemi = (G.coords[:, 2]>eps) + ((G.coords[:, 2]<eps)*(G.coords[:, 2]>-eps)*( G.coords[:, 1]>eps)) + ((G.coords[:, 2]<eps)*(G.coords[:, 2]>-eps)*(G.coords[:, 1]<eps)*(G.coords[:, 1]>-eps)*(G.coords[:, 0]>eps))
            coords = G.coords[index_north_hemi]
            assert coords.shape[0] == G.coords.shape[0]//2
        else:
            coords = G.coords
        self.sampling = Sampling(coords, sh_degree)
        print(f'Sampling number SHC: {self.sampling.S2SH.shape[1]}')
        assert self.sampling.S2SH.shape[1] == (sh_degree+1)*(sh_degree//2+1)
        
        self.laps, self.vec = self.get_healpix_laplacians(n_side, depth, laplacian_type="normalized", neighbor=8, pooling_name=pooling_name, hemisphere=hemisphere, legacy=legacy)
        self.pooling, self.patch_size_list = self.get_healpix_poolings(depth, pooling_mode, patch_size, n_side, pooling_name, hemisphere=hemisphere)
        print('-'*50)
    
    def get_healpix_laplacians(self, starting_nside, depth, laplacian_type, neighbor=8, pooling_name='mixed', hemisphere=False, legacy=True):
        """Get the healpix laplacian list for a certain depth.
        Args:
            starting_nside (int): initial healpix grid resolution.
            depth (int): the depth of the UNet.
            laplacian_type ["combinatorial", "normalized"]: the type of the laplacian.
        Returns:
            laps (list): increasing list of laplacians from smallest to largest resolution
        """
        print(f'Create Laplacians')
        laps = []
        vec = []
        if not pooling_name in ['spatial', 'spatial_vec', 'spatial_sh']:
            for i in range(depth):
                n_side = max(starting_nside//(2**i), 1) # Get resolution of the grid at depth i
                if n_side>0:
                    if legacy:
                        raise NotImplementedError('Legacy is not implemented')
                    else:
                        G = SphereHealpix(n_side, nest=True, k=neighbor) # Construct Healpix Graph at resolution n_side
                else:
                    if legacy:
                        raise NotImplementedError('Legacy is not implemented')
                    else:
                        G = SphereHealpix(n_side, nest=True, k=8) # Construct Healpix Graph at resolution n_side
                G.compute_laplacian(laplacian_type) # Compute Healpix laplacian
                laplacian = prepare_laplacian(G.L) # Get Healpix laplacian
                coords = G.coords
                if hemisphere:
                    eps = 1e-10
                    index_north_hemi = (G.coords[:, 2]>eps) + ((G.coords[:, 2]<eps)*(G.coords[:, 2]>-eps)*( G.coords[:, 1]>eps)) + ((G.coords[:, 2]<eps)*(G.coords[:, 2]>-eps)*(G.coords[:, 1]<eps)*(G.coords[:, 1]>-eps)*(G.coords[:, 0]>eps))
                    distance = G.coords.dot(G.coords.T)
                    indx = np.arange(distance.shape[0])[None].repeat(distance.shape[0], axis=0)
                    sel = distance < -1+1e-5
                    assert np.sum(np.sum(sel, axis=-1) != 1) == 0
                    match_sel = indx[sel]
                    laplacian = laplacian[index_north_hemi][:, index_north_hemi] + laplacian[index_north_hemi][:, match_sel][:, index_north_hemi]
                    coords = coords[index_north_hemi]
                laplacian = scipy_csr_to_sparse_tensor(laplacian)
                laps.append(laplacian)
                vec.append(coords)
                print(f'Laplacian at depth {i}: {laplacian.shape} and coordinates: {coords.shape}')
        elif pooling_name in ['spatial', 'spatial_vec', 'spatial_sh']:
            n_side = starting_nside
            if n_side>0:
                if legacy:
                    raise NotImplementedError('Legacy is not implemented')
                else:
                    G = SphereHealpix(n_side, nest=True, k=neighbor) # Construct Healpix Graph at resolution n_side
            else:
                if legacy:
                    raise NotImplementedError('Legacy is not implemented')
                else:
                    G = SphereHealpix(n_side, nest=True, k=8) # Construct Healpix Graph at resolution n_side
            G.compute_laplacian(laplacian_type) # Compute Healpix laplacian
            laplacian = prepare_laplacian(G.L) # Get Healpix laplacian
            coords = G.coords
            if hemisphere:
                eps = 1e-10
                index_north_hemi = (G.coords[:, 2]>eps) + ((G.coords[:, 2]<eps)*(G.coords[:, 2]>-eps)*( G.coords[:, 1]>eps)) + ((G.coords[:, 2]<eps)*(G.coords[:, 2]>-eps)*(G.coords[:, 1]<eps)*(G.coords[:, 1]>-eps)*(G.coords[:, 0]>eps))
                distance = G.coords.dot(G.coords.T)
                indx = np.arange(distance.shape[0])[None].repeat(distance.shape[0], axis=0)
                sel = distance < -1+1e-5
                assert np.sum(np.sum(sel, axis=-1) != 1) == 0
                match_sel = indx[sel]
                laplacian = laplacian[index_north_hemi][:, index_north_hemi] + laplacian[index_north_hemi][:, match_sel][:, index_north_hemi]
                coords = coords[index_north_hemi]
            laplacian = scipy_csr_to_sparse_tensor(laplacian)
            for i in range(depth):
                laps.append(laplacian)
                vec.append(coords)
                print(f'Laplacian at depth {i}: {laplacian.shape} and coordinates: {coords.shape}')
        else:
            for i in range(depth):
                laps.append(torch.ones(1, 1))
                vec.append(np.array([[1, 0, 0]]))
                print(f'Laplacian at depth {i}: {laplacian.shape} and coordinates: {coords.shape}')
        return laps[::-1], vec[::-1]

    def get_healpix_poolings(self, depth, pooling_mode, patch_size, n_side, pooling_name, hemisphere=False):
        """Get the healpix laplacian list for a certain depth.
        Args:
            starting_nside (int): initial healpix grid resolution.
            depth (int): the depth of the UNet.
            pooling_mode (str): specify the mode for pooling/unpooling. Can be max or average. Defaults to 'average'.
        Returns:
            laps (list): increasing list of laplacians from smallest to largest resolution
        """
        print('Create Poolings')
        print(f'Initial size: Patch size: {patch_size} - Resolution: {n_side}')
        poolings = []
        patch_size_list = []
        if pooling_name in ['mixed', 'bekkers']:
            for depth_i in range(depth-1):
                patch_size_list.append(patch_size)
                if hemisphere and n_side!=1:
                    ########### HR ###########
                    G = SphereHealpix(n_side, nest=True, k=8) # Construct Healpix Graph at resolution n_side
                    eps = 1e-10
                    index_north_hemi_hr = (G.coords[:, 2]>eps) + ((G.coords[:, 2]<eps)*(G.coords[:, 2]>-eps)*( G.coords[:, 1]>eps)) + ((G.coords[:, 2]<eps)*(G.coords[:, 2]>-eps)*(G.coords[:, 1]<eps)*(G.coords[:, 1]>-eps)*(G.coords[:, 0]>eps))
                    distance = G.coords.dot(G.coords.T)
                    indx = np.arange(distance.shape[0])[None].repeat(distance.shape[0], axis=0)
                    sel = distance < -1+1e-5
                    assert np.sum(np.sum(sel, axis=-1) != 1) == 0
                    match_sel_hr = indx[sel]
                    ########## LR ##############
                    G = SphereHealpix(n_side // 2, nest=True, k=8) # Construct Healpix Graph at resolution n_side
                    eps = 1e-10
                    index_north_hemi_lr = (G.coords[:, 2]>eps) + ((G.coords[:, 2]<eps)*(G.coords[:, 2]>-eps)*( G.coords[:, 1]>eps)) + ((G.coords[:, 2]<eps)*(G.coords[:, 2]>-eps)*(G.coords[:, 1]<eps)*(G.coords[:, 1]>-eps)*(G.coords[:, 0]>eps))
                    distance = G.coords.dot(G.coords.T)
                    indx = np.arange(distance.shape[0])[None].repeat(distance.shape[0], axis=0)
                    sel = distance < -1+1e-5
                    assert np.sum(np.sum(sel, axis=-1) != 1) == 0
                    match_sel_lr = indx[sel]
                else:
                    index_north_hemi_hr, index_north_hemi_lr, match_sel_hr, match_sel_lr = None, None, None, None
                if patch_size==1 and n_side==1:
                    pool = IdentityPooling()
                elif patch_size != 1 and n_side!=1:
                    print(patch_size, ' - ', n_side)
                    stride = (-(patch_size%2) + 2, -(patch_size%2) + 2, -(patch_size%2) + 2)
                    kernel_size_spa = ((patch_size>1) + 1, (patch_size>1) + 1, (patch_size>1) + 1)
                    pool = MixedPooling(mode=pooling_mode, kernel_size_spa=kernel_size_spa, stride=stride, hemisphere=hemisphere, index_north_hemi_hr=index_north_hemi_hr, index_north_hemi_lr=index_north_hemi_lr, match_sel_hr=match_sel_hr, match_sel_lr=match_sel_lr)
                    patch_size = int(((patch_size - 2) / 2) * (patch_size%2) + patch_size / 2)
                    n_side = n_side // 2
                elif patch_size==1:
                    pool = HealpixPooling(mode=pooling_mode, hemisphere=hemisphere, index_north_hemi_hr=index_north_hemi_hr, index_north_hemi_lr=index_north_hemi_lr, match_sel_hr=match_sel_hr, match_sel_lr=match_sel_lr)
                    n_side = n_side // 2
                else:
                    pool = SpatialPooling(mode=pooling_mode)
                    patch_size = int(((patch_size - 2) / 2) * (patch_size%2) + patch_size / 2)
                poolings.append(pool)
                print(f'Pooling after depth {depth_i}: {pool} - Patch size: {patch_size} - Resolution: {n_side}')
        elif pooling_name in ['spherical']:
            for depth_i in range(depth-1):
                patch_size_list.append(patch_size)
                if n_side!=1:
                    if hemisphere:
                        ########### HR ###########
                        G = SphereHealpix(n_side, nest=True, k=8) # Construct Healpix Graph at resolution n_side
                        eps = 1e-10
                        index_north_hemi_hr = (G.coords[:, 2]>eps) + ((G.coords[:, 2]<eps)*(G.coords[:, 2]>-eps)*( G.coords[:, 1]>eps)) + ((G.coords[:, 2]<eps)*(G.coords[:, 2]>-eps)*(G.coords[:, 1]<eps)*(G.coords[:, 1]>-eps)*(G.coords[:, 0]>eps))
                        distance = G.coords.dot(G.coords.T)
                        indx = np.arange(distance.shape[0])[None].repeat(distance.shape[0], axis=0)
                        sel = distance < -1+1e-5
                        assert np.sum(np.sum(sel, axis=-1) != 1) == 0
                        match_sel_hr = indx[sel]
                        ########## LR ##############
                        G = SphereHealpix(n_side // 2, nest=True, k=8) # Construct Healpix Graph at resolution n_side
                        eps = 1e-10
                        index_north_hemi_lr = (G.coords[:, 2]>eps) + ((G.coords[:, 2]<eps)*(G.coords[:, 2]>-eps)*( G.coords[:, 1]>eps)) + ((G.coords[:, 2]<eps)*(G.coords[:, 2]>-eps)*(G.coords[:, 1]<eps)*(G.coords[:, 1]>-eps)*(G.coords[:, 0]>eps))
                        distance = G.coords.dot(G.coords.T)
                        indx = np.arange(distance.shape[0])[None].repeat(distance.shape[0], axis=0)
                        sel = distance < -1+1e-5
                        assert np.sum(np.sum(sel, axis=-1) != 1) == 0
                        match_sel_lr = indx[sel]
                    else:
                        index_north_hemi_hr, index_north_hemi_lr, match_sel_hr, match_sel_lr = None, None, None, None
                    pool = HealpixPooling(mode=pooling_mode, hemisphere=hemisphere, index_north_hemi_hr=index_north_hemi_hr, index_north_hemi_lr=index_north_hemi_lr, match_sel_hr=match_sel_hr, match_sel_lr=match_sel_lr)
                    n_side = n_side // 2
                else:
                    pool = IdentityPooling()
                poolings.append(pool)
                print(f'Pooling after depth {depth_i}: {pool} - Patch size: {patch_size} - Resolution: {n_side}')
        elif pooling_name in ['spatial', 'spatial_vec', 'spatial_sh']:
            for depth_i in range(depth-1):
                patch_size_list.append(patch_size)
                print(patch_size_list)
                if patch_size!=1:
                    stride = (-(patch_size%2) + 2, -(patch_size%2) + 2, -(patch_size%2) + 2)
                    kernel_size_spa = ((patch_size>1) + 1, (patch_size>1) + 1, (patch_size>1) + 1)
                    pool = SpatialPooling(mode=pooling_mode, kernel_size_spa=kernel_size_spa, stride=stride)    
                    patch_size = int(((patch_size - 2) / 2) * (patch_size%2) + patch_size / 2)
                else:
                    pool = IdentityPooling()
                poolings.append(pool)
                print(f'Pooling after depth {depth_i}: {pool} - Patch size: {patch_size} - Resolution: {n_side}')
        elif pooling_name=='muller':
            patch_size_list.append(patch_size)
            for depth_i in range(depth-1):
                pool = IdentityPooling()
                poolings.append(pool)
                print(f'Pooling after depth {depth_i}: {pool} - Patch size: {patch_size} - Resolution: {n_side}')
        patch_size_list.append(patch_size)
        return poolings[::-1], patch_size_list[::-1]
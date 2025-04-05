# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:31:40 2021

@author: Philipp Krah

Definition of the class Transform that models a transformation applied to the
field.
"""
# ============================================================================ #
#                              MODULES IMPORTATION                             #
# ============================================================================ #
from scipy import sparse
from numpy import meshgrid, size, reshape, floor
import numpy as np
import scipy.ndimage as ndimage
#from numba import njit
# ============================================================================ #


# ---------------------------------------------------------------------------- #
#                             Lagrange interpolation                           #
# ---------------------------------------------------------------------------- #
def lagrange(xvals, xgrid, j):
    """
    Returns the j-th basis polynomial evaluated at xvals using the grid points
    listed in xgrid.
    """
    xgrid = np.asarray(xgrid)
    if not isinstance(xvals,list):
        xvals=[xvals]
    n = np.size(xvals)
    Lj = np.zeros(n)
    for i,xval in enumerate([xvals]):
        nominator = xval - xgrid
        denominator = xgrid[j] - xgrid
        p = nominator/(denominator+np.finfo(float).eps)
        p[j] = 1
        Lj[i] = np.prod(p)

    return Lj

#@njit()
def lagrange_numba(xvals, xgrid, j):
    """
    Returns the j-th basis polynomial evaluated at xvals using the grid points
    listed in xgrid.
    """
    xgrid = np.asarray(xgrid)
    for i,xval in enumerate([xvals]):
        nominator = xval - xgrid
        denominator = xgrid[j] - xgrid
        p = nominator/(denominator+np.finfo(float).eps)
        p[j] = 1
        Lj = np.prod(p)

    return Lj

#@njit()
def meshgrid2D(x, y):
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for j in range(y.size):
        for k in range(x.size):
            xx[j,k] = x[k]
            yy[j,k] = y[j]
    return yy, xx

#@njit()
def compute_general_shift_matrix_numba( shifts, domain_size, spacings, Ngrid, Ix, Iy):
    """
    interpolation scheme        lagrange_idx(x)= (x-x_{idx-1})/(x_idx - x_0)+
    -1      0   x    1       2                    ...+(x-x_{idx+2})/(x_idx - x_{idx+2})
     +      +   x    +       +
    idx_m1  idx_0    idx_1   idx_2
    =idx_0-1        =idx_0+1

    :param shifts: shift(x_i,t_j) assumes an array of size i=0,...,Nx -1 ; j=0,...,Nt
    :param domain_length:
    :param spacings:
    :param Npoints:
    :return:
    """
    col = [np.int(x) for x in range(0)]
    row = [np.int(x) for x in range(0)]
    val = [np.float(x) for x in range(0)]
    dx, dy = spacings
    Nx, Ny = Ngrid
    for ik in range(Nx*Ny):
        # define shifts
        shift_x = shifts[0,ik] # delta_1(x_i,y_i,t)
        shift_x = np.mod(shift_x,domain_size[0]) # if periodicity is assumed
        shift_y = shifts[1,ik] # delta_2(x_i,y_i,t)
        shift_y = np.mod(shift_y,domain_size[1]) # if periodicity is assumed
        # lexicographical index ik to local grid index ix, iy
        (ix, iy) = (Ix[ik], Iy[ik])

        # shift is close to sogenerame discrete index:
        idx_0 = floor(shift_x / dx)
        idy_0 = floor(shift_y / dy)
        # save all neighbours
        idx_list = np.asarray([idx_0 - 1, idx_0, idx_0 + 1, idx_0 + 2], dtype=np.int32) + ix
        idx_list = np.asarray([np.mod(idx, Nx) for idx in idx_list]) # assumes periodicity
        idy_list = np.asarray([idy_0 - 1, idy_0, idy_0 + 1, idy_0 + 2], dtype=np.int32) + iy
        idy_list = np.asarray([np.mod(idy, Ny) for idy in idy_list])  # assumes periodicity
        # compute the distance to the index
        delta_idx = shift_x / dx - idx_0
        delta_idy = shift_y / dy - idy_0
        # compute the 4 langrage basis elements
        lagrange_coefs_x = np.array([lagrange_numba(delta_idx, [-1, 0, 1, 2], j) for j in range(4)])
        lagrange_coefs_y = np.array([lagrange_numba(delta_idy, [-1, 0, 1, 2], j) for j in range(4)])
        lagrange_coefs = np.outer(lagrange_coefs_y, lagrange_coefs_x)
        lagrange_coefs = lagrange_coefs.reshape((-1,))
        #
        IDX_mesh, IDY_mesh = meshgrid2D(idx_list, idy_list)
        IK_mesh = IDX_mesh + IDY_mesh*Nx
        IKs = IK_mesh.reshape((-1,))
        col += list(IKs)
        row += list(ik * np.ones_like(IKs))
        val += list(lagrange_coefs)
        # if idx_list[0] < 0 : idx_list[0] += Npoints
        # if idx_list[2] > Npoints - 1: idx_list[2] -= Npoints
        # if idx_list[3] > Npoints - 1: idx_list[3] -= Npoints

    val = np.asarray(val)
    return col,row,val


class Transform:
    """
    This class implements a transformation associated to a frame.
    A transformation can be implemented as a
    + shift: T^c q(x,t) = q(x-ct,t)
    + rotation: T^w q(x,y,t) = q(R(omega)(x,y),t) where R(omega) is the
    rotation matrix

    :param Ngrid: 
    :type Ngrid: List[int]

    :param Nvar: 
    :type Nvar: int

    :param Ntime: 
    :type Ntime: int
    
    :param data_shape: 
    :type data_shape: List[int]

    :param domain_size: 
    :type domain_size: List[int]

    :param transfo_type: 
    :type transfo_type: str

    :param dim: 
    :type dim: int

    :param interp_order: 
    :type interp_order: int

    :param dx: 
    :type dx: List[:class:`numpy.float64`]
    """
    def __init__(self, data_shape, domain_size, transfo_type="shift",
                 shifts=None, dx=None, rotations=None, rotation_center=None,
                 use_scipy_transform=False, interp_order=3):
        self.Ngrid = data_shape[:2]
        self.Nvar = data_shape[2]
        self.Ntime = data_shape[3]
        self.data_shape = data_shape
        self.domain_size = domain_size
        self.transfo_type = transfo_type
        self.dim = size(dx)
        self.interp_order = interp_order
        self.dx = dx  # list of lattice spacings
        if self.dim == 1:
            self.shifts_pos, self.shifts_neg = self.init_shifts_1D(dx[0],
                                                                   domain_size[0],
                                                                   self.Ngrid[0],
                                                                   shifts[:],
                                                                   Nvar=self.Nvar)
            self.shift = self.shift1
        else:
            if transfo_type == "shift":
                if use_scipy_transform:
                    self.shift = self.shift_scipy
                    self.shifts_pos =  shifts    # dim x Ntime shiftarray (one element for one time instance)
                    one = np.ones_like(shifts)
                    one[0,:] = dx[0]
                    one[1,:] = 0  #dx[1]
                    self.shifts_neg = -shifts  # dim x Ntime shiftarray (one element for one time instance)
                else: # own implementation for shifts: is much faster then ndimage
                    self.shifts_pos, self.shifts_neg = self.init_shifts_2D(dx, domain_size, self.Ngrid, shifts, Nvar= self.Nvar)
                    self.shift = self.shift1
            if transfo_type == "rotation":
                assert(size(dx)==2), "is only implemented for spatial fields in 2 dimensions"
                #assert(np.sum(rotation_center)==0), "rotation center should be in the middle!"
                self.rotations = rotations # is an array with [omega(t_1), ... omega(t_Ntime)] rotation angles at different timepoints
                self.rotation_center = rotation_center # array including center of rotation(x_0, y_0)
            if transfo_type == "shiftRot":
                assert(size(dx)==2), "is only implemented for spatial fields in 2 dimensions"
                if use_scipy_transform:
                    self.shift = self.shift_scipy
                    self.shifts_pos =  shifts    # dim x Ntime shiftarray (one element for one time instance)
                    self.shifts_neg = -shifts  # dim x Ntime shiftarray (one element for one time instance)
                else: # own implementation for shifts: is much faster then ndimage
                    self.shifts_pos, self.shifts_neg = self.init_shifts_2D(dx, domain_size, self.Ngrid, shifts, Nvar= self.Nvar)
                    self.shift = self.shift1
                self.rotations = rotations
                self.rotation_center = rotation_center


    def apply(self, field):
        """
        This function returns the shifted field.

        .. math::

            q(x-s,t) = T^{s}[q(x,t)]

        Here, the shift is simply :math:`s=ct`
        In the default case where c= ~velocity of the frame~, the field is
        shifted back to the original frame.
        (You may call it the laboratory frame)
        Before we perform the shift, the frame has to be put together in
        the co-moving frame. This can be done by build_field().

        :param field: 
        :type field:

        :return: 
        :rtype: 
        """
        input_shape = np.shape(field)
        field = reshape(field,self.data_shape)
        if self.transfo_type=="shift":
            ftrans = self.shift(field, self.shifts_pos)
        elif self.transfo_type == "rotation":
            ftrans = self.rotate(field, self.rotations)
        elif self.transfo_type == "shiftRot":
            # ~ auxField = self.shift(field,self.shiftMatrices_pos)         #shift to origin
            field = self.rotate(field,self.rotations)                 #rotate and return
            ftrans = self.shift(field,self.shifts_pos)                   #shift to origin
        elif self.transfo_type == "identity":
            ftrans = field
        else:
            print("Transformation type: {} not known".format(self.transfo_type))

        return reshape(ftrans, input_shape)

    def reverse(self, field):
        """
        """
        input_shape = np.shape(field)
        field = reshape(field, self.data_shape)
        if self.transfo_type=="shift":
            ftrans = self.shift(field,self.shifts_neg)
        elif self.transfo_type == "rotation":
            ftrans = self.rotate(field, -self.rotations)
        elif self.transfo_type == "shiftRot":
            field = self.shift(field,self.shifts_neg)                   #shift back and return
            ftrans = self.rotate(field,-self.rotations)               #rotate back
            # ~ return self.shift(auxField,self.shiftMatrices_neg)          #shift back and return
        elif self.transfo_type == "identity":
            ftrans = field
        else:
            print("Transformation type: %s not known"%self.transfo_type)

        return reshape(ftrans, input_shape)

    def shift1(self, field, shifts):
        """
        This function returns the shifted field.
        $q(x-s,t)=T^s[q(x,t)]$
        here the shift is simply s=c*t
        In the default case where c= ~velocity of the frame~,
        the field is shifted back to the original frame.
        (You may call it the laboratory frame)
        Before we shift the frame has to be put togehter in the co-moving
        frame. This can be done by build_field().
        """
        Ntime = np.size(field,-1)
        field_shift = np.zeros_like(field)
        for it in range(Ntime):
            vec = np.reshape(field[...,it],-1)
            vec_shift = shifts[it]@vec
            field_shift[...,it] = np.reshape(vec_shift,self.data_shape[:-1])

        return field_shift
        
    def shift_scipy(self,field,shifts):
        """
        This function returns the shifted field.
        $q(x-s,t)=T^s[q(x,t)]$ 
        here the shift is simply s=c*t
        In the default case where c= ~velocity of the frame~,
        the field is shifted back to the original frame. 
        (You may call it the laboratory frame)
        Before we shift the frame has to be put togehter in the co-moving
        frame. This can be done by build_field().
        """
        input_shape = np.shape(field)
        Ntime = np.size(field,-1)
        field_shift = np.zeros([*self.Ngrid,Ntime])
        for it in range(Ntime):

            #DeltaS = -np.round(np.divide(shifts[:, it], self.dx))
            DeltaS = -np.divide(shifts[:, it], self.dx)
            q = np.reshape(field[...,it], self.Ngrid)
            #field_shift[...,it] = np.roll(q, int(DeltaS[1]), axis=1)
            #field_shift[..., it] = np.roll(q, DeltaS[0], axis=1)
            #q = np.reshape(field[...,it], self.Ngrid)
            field_shift[...,it] = ndimage.shift(q,DeltaS,mode='grid-wrap')
        return np.reshape(field_shift,input_shape)
    # Note (MI): the shifts need to be scaled w.r.t the image and are reversed 


    def rotate(self, field, rotations):
        input_shape = np.shape(field)
        Ntime = np.size(field,-1)
        field_rot = np.zeros([*self.Ngrid,Ntime])
        for it in range(Ntime):
            angle = rotations[it]
            q = np.reshape(field[...,it], self.Ngrid)
            field_rot[...,it] = ndimage.rotate(q, angle*180.0/np.pi, reshape=False)
            
        return np.reshape(field_rot,input_shape)


    def init_shifts_2D(self, dX, domain_size, Ngrid, shifts, Nvar = 1):
        ### implement pos shift matrix ###
        print("init shift matrix ...")

        if not isinstance(self.interp_order, list):
            interp_order = [self.interp_order, self.interp_order]
        else:
            interp_order = self.interp_order

        print("Setting up the shift matrices, with interpolation order:")
        print("Forward T^k:     O(h^%d)" % interp_order[0])
        print("Backward T^(-k): O(h^%d)" % interp_order[1])

        Nx, Ny = Ngrid
        Lx, Ly = domain_size
        dx, dy = dX
        Nt = np.size(shifts,-1)

        shift_pos_mat_list = []
        shift_neg_mat_list = []

        if np.ndim(shifts) == 2:
            # this assumes that the shifts are independent variables in x and y
            # Hence, the shifts can be written in the following form: delta(x,y,t) = delta_1(x,t)delta_2(y,t).
            # Note for general grid transformations, this must not be the case, since the shift can locally depend on both variables (x,y)!
            # This is considered in the "else" case
            shiftx_pos = shifts[0,...]
            shifty_pos = shifts[1,...]


            shiftx_pos_mat_list = self.compute_shift_matrix(shiftx_pos, Lx, dx,
                                                            Nx, order = interp_order[0])
            shifty_pos_mat_list = self.compute_shift_matrix(shifty_pos, Ly, dy,
                                                            Ny, order = interp_order[0])

            # kron for each time slice
            for shiftx, shifty in zip(shiftx_pos_mat_list, shifty_pos_mat_list):
                shift_pos_mat_list.append(sparse.kron(sparse.kron(shiftx, shifty),
                                                      sparse.eye(Nvar)))

            ### implement neg shift matrix ###


            shiftx_neg = -shiftx_pos
            shifty_neg = -shifty_pos
            shiftx_neg_mat_list = self.compute_shift_matrix(shiftx_neg, Lx, dx,
                                                            Nx, order = interp_order[1])
            shifty_neg_mat_list = self.compute_shift_matrix(shifty_neg, Ly, dy,
                                                            Ny, order = interp_order[1])

            # kron for each time slice
            for shiftx, shifty in zip(shiftx_neg_mat_list, shifty_neg_mat_list):
                shift_neg_mat_list.append(sparse.kron(sparse.kron(shiftx, shifty),
                                                      sparse.eye(Nvar)))

        else:

            shift_pos_list = self.compute_general_shift_matrix(shifts, domain_size, [dx,dy], Ngrid)
            for shiftmat in shift_pos_list:
                shift_pos_mat_list.append(sparse.kron(shiftmat, sparse.eye(Nvar)))
            # shifts[0, ...] -= dx
            # shifts[1, ...] -= dy
            shift_neg_list = self.compute_general_shift_matrix(-shifts, domain_size, [dx,dy], Ngrid)
            for shiftmat in shift_neg_list:
                shift_neg_mat_list.append(sparse.kron(shiftmat, sparse.eye(Nvar)))

        #
        # shiftx_pos = shifts[0,...]
        # shifty_pos = shifts[1,...]

        return shift_pos_mat_list, shift_neg_mat_list

    def init_shifts_1D(self, dx, Lx, Nx, shifts, Nvar=1):
        # If the interp_order is a list we take the first value for the forward transform T^k
        # and the second element for the backward transform T^{-k}
        # This property is used in the sPOD-J2 algorithm, since the redistribution/projection step allows lower
        # interpolation order in the backward transform, without loosing precision.
        if not isinstance(self.interp_order, list):
            interp_order = [self.interp_order, self.interp_order]
        else:
            interp_order = self.interp_order

        print("Setting up the shift matrices, with interpolation order:")
        print("Forward T^k:     O(h^%d)"%interp_order[0])
        print("Backward T^(-k): O(h^%d)" % interp_order[1])
        ### implement pos shift matrix ###
        shift_pos_mat_list = []
        shiftx_pos = shifts[:]
        shiftx_pos_mat_list = self.compute_shift_matrix(shiftx_pos, Lx, dx, Nx,
                                                        order = interp_order[0])
        # kron for each time slice
        for shiftx in shiftx_pos_mat_list:
            shift_pos_mat_list.append(sparse.kron(shiftx, sparse.eye(Nvar)))

        ### implement neg shift matrix ###
        shift_neg_mat_list = []
        shiftx_neg = -shiftx_pos
        shiftx_neg_mat_list = self.compute_shift_matrix(shiftx_neg, Lx, dx, Nx,
                                                        order = interp_order[1])

        # kron for each time slice
        for shiftx in shiftx_neg_mat_list:
            shift_neg_mat_list.append(sparse.kron(shiftx, sparse.eye(Nvar)))

        return shift_pos_mat_list, shift_neg_mat_list
    
    def init_rotations(self):
        ### implement pos shift matrix ###
        rotation_pos_mat_list = []
                
        Nx, Ny = self.Ngrid
        Lx, Ly = self.domain_size
        dx, dy = self.dx
        
        rotations_pos = self.rotations
        rot_center    = self.rotation_center
        rotations_neg = - rotations_pos
        
        rotation_pos_mat_list=self.compute_rotation_matrix(rotations_pos, rot_center, \
                                                         self.domain_size, self.dx, self.Ngrid)
        
        rotation_neg_mat_list=self.compute_rotation_matrix(rotations_neg, rot_center, \
                                                         self.domain_size, self.dx, self.Ngrid)
        
        return rotation_pos_mat_list, rotation_neg_mat_list
        
    def compute_shift_matrix(self,shift_list, domain_length, spacing, Npoints, order=3):
        Mat = []
        domain_size = self.domain_size
        for it,shift in enumerate(shift_list):
            
            # we assume periodicity here
            shift = np.mod(shift, domain_size[0])  # if periodicity is assumed

            ''' interpolation scheme        lagrange_idx(x)= (x-x_{idx-1})/(x_idx - x_0)+
            -1      0   x    1       2                    ...+(x-x_{idx+2})/(x_idx - x_{idx+2})
             +      +   x    +       +
           idx_m1  idx_0    idx_1   idx_2
          =idx_0-1        =idx_0+1
           '''
           
            # shift is close to some discrete index:
            idx_0 = np.floor(shift/spacing)
            # save all neighbours
            if order == 5:
                idx_list = np.asarray([idx_0 - 2,idx_0 - 1, idx_0, idx_0 + 1, idx_0 + 2, idx_0 + 3], dtype=np.int32)
            elif order == 3:
                idx_list = np.asarray([idx_0 - 1, idx_0, idx_0 + 1, idx_0 + 2], dtype=np.int32)
            elif order == 1:
                idx_list = np.asarray([ idx_0, idx_0 + 1], dtype=np.int32)
            else:
                assert(False), "please choose correct order for interpolation"

            idx_list = np.asarray([np.mod(idx, Npoints) for idx in idx_list])  # assumes periodicity
            # subdiagonals needed if point is on other side of domain
            idx_subdiags_list = idx_list - Npoints
            # compute the distance to the index
            delta_idx = shift/spacing - idx_0
            # compute the 4 langrage basis elements
            if order == 5:
                lagrange_coefs = [lagrange(delta_idx, [-2,-1, 0, 1, 2, 3], j) for j in range(6)]
            elif order == 3:
                lagrange_coefs = [lagrange(delta_idx, [-1,0,1,2], j) for j in range(4)]
            elif order == 1:
                lagrange_coefs = [lagrange(delta_idx, [0, 1], j) for j in range(2)]
            else:
                assert (False), "please choose correct order for interpolation"
            # for the subdiagonals as well
            lagrange_coefs = lagrange_coefs + lagrange_coefs
            
            # band diagonals for the shift matrix
            offsets = np.concatenate([idx_list,idx_subdiags_list])
            diagonals = [np.ones(Npoints+1)*Lj  for Lj in lagrange_coefs]

            Mat.append(sparse.diags(diagonals,offsets,shape=[Npoints,Npoints]))
        
        return Mat


    def compute_general_shift_matrix(self, shifts, domain_size, spacings, Ngrid):
        """

        :param shifts: shift(x_i,t_j) assumes an array of size i=0,...,Nx -1 ; j=0,...,Nt
        :param domain_length:
        :param spacings:
        :param Npoints:
        :return:
        """

        ''' interpolation scheme        lagrange_idx(x)= (x-x_{idx-1})/(x_idx - x_0)+
        -1      0   x    1       2                    ...+(x-x_{idx+2})/(x_idx - x_{idx+2})
         +      +   x    +       +
        idx_m1  idx_0    idx_1   idx_2
        =idx_0-1        =idx_0+1
       '''
        from joblib import Parallel, delayed
        # Function has to be parallelized since it take a long time to set up a
        # general transformation matrix

        Nt = np.size(shifts, -1)
        Nx, Ny = Ngrid
        [Ix, Iy] = meshgrid2D(np.arange(0, Nx), np.arange(0, Ny))
        Ix, Iy = Ix.flatten(), Iy.flatten()

        def my_parallel_fun(it):
            col,row,val = compute_general_shift_matrix_numba(shifts[..., it], np.asarray(domain_size), np.asarray(spacings),
                                               np.asarray(Ngrid), Ix, Iy)
            return sparse.coo_matrix((val, (row, col)), shape=(Nx * Ny, Nx * Ny)).tocsc()

        Tmat_time_list = Parallel(n_jobs=-2)(delayed(my_parallel_fun)(it) for it in range(Nt))
        # for it in range(Nt):
        #     Tmat_time_list.append(sparse.coo_matrix((val, (row, col)), shape=(Nx * Ny, Nx * Ny)))

        return Tmat_time_list


    # def compute_rotation_matrix(self, rotations, center, domain_length, spacing, Npoints):
    #     """

    #     Parameters
    #     ----------
    #     rotations : np.array of rotations size [#Snapshots]
    #                 This array should include the angle to the reference position
    #                 at every time point
    #     center : np.array of size 2
    #              (x,y) position of the center of rotation
    #     domain_length : np.array of size 2
                
    #     spacing : np.array of size 2
    #             lattice spacing
    #     Npoints : np.array Number of Gridpoints in x and y direction
            

    #     Returns
    #     -------
    #     Mat : Array of sparse matrix which rotate the field arround a center
    #          q(x',y', t) = M(omega(t))@ q(x,y,t) where q is a snapshot reshaped to a columnvector
    #          here (x')  = (cos(omega)* x - sin(omega)* y )
    #               (y')  = (sin(omega)* x + cos(omega)* y )
    #         Since the data only exists on discrete time points, we have to interpolate!
    #         The interpolation stencil ist defined in M(omega(t))
    #     """
    #     Mat = []
    #     for shift in rotations:
            
    #         # we assume periodicity here
    #         if shift > domain_length:
    #             shift = shift - domain_length
    #         elif shift < 0:
    #             shift = shift + domain_length
        
    #         ''' interpolation scheme        lagrange_idx(x)= (x-x_{idx-1})/(x_idx - x_0)+
    #         -1      0   x    1       2                    ...+(x-x_{idx+2})/(x_idx - x_{idx+2})
    #          +      +   x    +       +
    #        idx_m1  idx_0    idx_1   idx_2
    #       =idx_0-1        =idx_0+1
    #        '''
           
    #         # shift is close to some discrete index:
    #         idx_0 = np.floor(shift/spacing)
    #         # save all neighbours
    #         idx_list = np.asarray([idx_0-1, idx_0, idx_0+1, idx_0+2],dtype=np.int32)
            
    #         if idx_list[0] < 0 : idx_list[0] += Npoints 
    #         if idx_list[3] > Npoints-1 : idx_list[3] -= Npoints
    #         # subdiagonals needed if point is on other side of domain
    #         idx_subdiags_list = idx_list - Npoints
    #         # compute the distance to the index
    #         delta_idx = shift/spacing - idx_0
    #         # compute the 4 langrage basis elements
    #         lagrange_coefs = [lagrange(delta_idx, [-1,0,1,2], j) for j in range(4)]
    #         # for the subdiagonals as well
    #         lagrange_coefs = lagrange_coefs +lagrange_coefs
            
    #         # band diagonals for the shift matrix
    #         offsets = np.concatenate([idx_list,idx_subdiags_list])
    #         diagonals = [np.ones(Npoints)*Lj  for Lj in lagrange_coefs]

   #         Mat.append(sparse.diags(diagonals,offsets,shape=[Npoints,Npoints]))
        
   #     return Mat

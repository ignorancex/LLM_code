from __future__ import print_function, division
import numpy as np

from nbodykit import CurrentMPIComm
from nbodykit.lab import MPI
from nbodykit.source.mesh.field import FieldMesh
from pmesh.pm import RealField, ComplexField

from cosmo_model import CosmoModel
from gen_cosmo_fcns import generate_calc_Da


def get_cstat(data, statistic, comm=None):
    """
    Compute a collective statistic across all ranks and return as float.
    Must be called by all ranks.
    """
    #if isinstance(data, MeshSource):
    #    data = data.compute().value
    if isinstance(data, RealField) or isinstance(data, ComplexField):
        data = data.value
    else:
        assert type(data) == np.ndarray
    if comm is None:
        from nbodykit import CurrentMPIComm
        comm = CurrentMPIComm.get()

    if statistic == 'min':
        return comm.allreduce(data.min(), op=MPI.MIN)
    elif statistic == 'max':
        return comm.allreduce(data.max(), op=MPI.MAX)
    elif statistic == 'mean':
        # compute the mean
        csum = comm.allreduce(data.sum(), op=MPI.SUM)
        csize = comm.allreduce(data.size, op=MPI.SUM)
        return csum / float(csize)
    elif statistic == 'rms':
        mean = get_cmean(data, comm=comm)
        rsum = comm.allreduce(((data-mean)**2).sum())
        csize = comm.allreduce(data.size)
        rms = (rsum / float(csize))**0.5
        return rms
    elif statistic == 'sum':
        csum = comm.allreduce(data.sum(), op=MPI.SUM)
        return csum
    elif statistic == 'sqsum':
        csqsum = comm.allreduce((data**2).sum(), op=MPI.SUM)
        return csqsum
    else:
        raise Exception("Invalid statistic %s" % statistic)


def get_cmean(data, comm=None):
    return get_cstat(data, 'mean', comm=comm)

def get_csum(data, comm=None):
    return get_cstat(data, 'sum', comm=comm)

def get_csqsum(data, comm=None):
    return get_cstat(data, 'sqsum', comm=comm)

def get_cmin(data, comm=None):
    return get_cstat(data, 'min', comm=comm)


def get_cmax(data, comm=None):
    return get_cstat(data, 'max', comm=comm)


def get_crms(data, comm=None):
    return get_cstat(data, 'rms', comm=comm)


def get_cstats_string(data, comm=None):
    """
    Get collective statistics (rms, min, mean, max) of data and return as string.
    Must be called by all ranks.
    """
    from collections import OrderedDict
    stat_names = ['rms', 'min', 'mean', 'max']
    cstats = OrderedDict()
    iscomplex = False
    for s in stat_names:
        cstats[s] = get_cstat(data, s)
        if np.iscomplex(cstats[s]):
            iscomplex = True

    if iscomplex:
        return 'rms, min, mean, max: %s %s %s %s' % (str(
            cstats['rms']), str(cstats['min']), str(
                cstats['mean']), str(cstats['max']))
    else:
        return 'rms, min, mean, max: %g %g %g %g' % (
            cstats['rms'], cstats['min'], cstats['mean'], cstats['max'])


def calc_quadratic_field(
        base_field_mesh=None,
        second_base_field_mesh=None,
        quadfield=None,
        smoothing_of_base_field=None,
        smoothing_of_second_base_field=None,
        #return_in_k_space=False,
        verbose=False):
    """
    Calculate quadratic field, essentially by squaring base_field_mesh
    with filters applied before squaring. 

    Parameters
    ----------
    base_field_mesh : MeshSource object, typically a FieldMesh object
        Input field that will be squared.

    second_base_field_mesh : MeshSource object, typically a FieldMesh object
        Use this to multiply two fields, e.g. delta1*delta2 or G2[delta1,delta2].
        Only implemented for tidal_G2 at the moment.

    quadfield : string
        Represents quadratic field to be calculated. Can be
        - 'tidal_s2': Get s^2 = 3/2*s_ij*s_ij = 3/2*[d_ij d_ij - 1/3 delta^2] 
                      = 3/2*d_ijd_ij - delta^2/2,
                      where s_ij = (k_ik_j/k^2-delta_ij^K/3)basefield(\vk) and
                      d_ij = k_ik_j/k^2*basefield(\vk).
        - 'tidal_G2': Get G2[delta] = d_ij d_ij - delta^2. This is orthogonal to
                      delta^2 at low k which can be useful; also see Assassi et al (2014).
        - 'shift': Get shift=\vPsi\cdot\vnabla\basefield(\vx), where vPsi=-ik/k^2*basefield(k).
        - 'PsiNablaDelta': Same as 'shift'
        - 'growth': Get delta^2(\vx)
        - 'F2': Get F2[delta] = 17/21 delta^2 + shift + 4/21 tidal_s2
                              = delta^2 + shift + 2/7 tidal_G2

    Returns
    -------
    Return the calculated (Ngrid,Ngrid,Ngrid) field as a FieldMesh object.

    """
    comm = CurrentMPIComm.get()

    if second_base_field_mesh is not None:
        if not quadfield.endswith('two_meshs'):
            raise Exception(
                'second_base_field_mesh not implemented for quadfield %s' 
                % quadfield)

    # apply smoothing
    if smoothing_of_base_field is not None:
        base_field_mesh = apply_smoothing(mesh_source=base_field_mesh,
                                          **smoothing_of_base_field)
        
    if second_base_field_mesh is not None:
        if smoothing_of_second_base_field is not None:
            second_base_field_mesh = apply_smoothing(mesh_source=second_base_field_mesh,
                                          **smoothing_of_second_base_field)


    # compute quadratic (or cubic) field
    if quadfield == 'growth':
        out_rfield = base_field_mesh.compute(mode='real')**2

    elif quadfield == 'growth_two_meshs':
        out_rfield = (
            base_field_mesh.compute(mode='real')
            *second_base_field_mesh.compute(mode='real'))

    elif quadfield == 'growth-mean':
        out_rfield = base_field_mesh.compute(mode='real')**2
        mymean = out_rfield.cmean()
        out_rfield -= mymean

    elif quadfield == 'cube-mean':
        out_rfield = base_field_mesh.compute(mode='real')**3
        mymean = out_rfield.cmean()
        out_rfield -= mymean

    elif quadfield == 'tidal_G2':
        # Get G2[delta] = d_ijd_ij - delta^2

        if second_base_field_mesh is None:

            # Compute -delta^2(\vx)
            out_rfield = -base_field_mesh.compute(mode='real')**2

            # Compute d_ij(x). It's symmetric in i<->j so only compute j>=i.
            # d_ij = k_ik_j/k^2*basefield(\vk).
            for idir in range(3):
                for jdir in range(idir, 3):

                    def my_transfer_function(k3vec, val, idir=idir, jdir=jdir):
                        kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                        kk[kk == 0] = 1
                        return k3vec[idir] * k3vec[jdir] / kk * val

                    dij_k = base_field_mesh.apply(my_transfer_function,
                                                  mode='complex',
                                                  kind='wavenumber')
                    del my_transfer_function
                    # do fft and convert field_mesh to RealField object
                    dij_x = dij_k.compute(mode='real')
                    if verbose:
                        rfield_print_info(dij_x, comm, 'd_%d%d: ' % (idir, jdir))

                    # Add \sum_{i,j=0..2} d_ij(\vx)d_ij(\vx)
                    #   = [d_00^2+d_11^2+d_22^2 + 2*(d_01^2+d_02^2+d_12^2)]
                    if jdir == idir:
                        fac = 1.0
                    else:
                        fac = 2.0
                    out_rfield += fac * dij_x**2
                    del dij_x, dij_k

        else:
            raise Exception('use tidal_G2_two_meshs')

    elif quadfield == 'tidal_G2_two_meshs':
        # use second_base_field_mesh
        # Compute -delta1*delta2(\vx)
        out_rfield = -(
            base_field_mesh.compute(mode='real')
            * second_base_field_mesh.compute(mode='real') )

        # Compute d_ij(x). It's not symmetric in i<->j so compute all i,j
        # d_ij = k_ik_j/k^2*basefield(\vk).
        for idir in range(3):
            for jdir in range(3):

                def my_transfer_function(k3vec, val, idir=idir, jdir=jdir):
                    kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                    kk[kk == 0] = 1
                    return k3vec[idir] * k3vec[jdir] / kk * val

                dij_k = base_field_mesh.apply(
                    my_transfer_function,
                    mode='complex',
                    kind='wavenumber')
                second_dij_k = second_base_field_mesh.apply(
                    my_transfer_function,
                    mode='complex',
                    kind='wavenumber')
                del my_transfer_function
                
                # do fft and convert field_mesh to RealField object
                dij_x = dij_k.compute(mode='real')
                second_dij_x = second_dij_k.compute(mode='real')
                if verbose:
                    rfield_print_info(
                        dij_x, comm, 'd_%d%d: ' % (idir, jdir))
                    rfield_print_info(
                        second_dij_x, comm, 'd_%d%d: ' % (idir, jdir))

                # Add \sum_{i,j=0..2} d_ij(\vx)second_d_ij(\vx)
                # 29 Jul 2020: Had bug before, assumign symmetry.
                out_rfield += dij_x * second_dij_x

                del dij_x, dij_k, second_dij_x, second_dij_k


    elif quadfield == 'tidal_s2':
        # Get s^2 = 3/2*d_ijd_ij - delta^2/2
        # Compute -delta^2(\vx)/2
        out_rfield = -base_field_mesh.compute(mode='real')**2 / 2.0

        # Compute d_ij(x). It's symmetric in i<->j so only compute j>=i.
        # d_ij = k_ik_j/k^2*basefield(\vk).
        for idir in range(3):
            for jdir in range(idir, 3):

                def my_transfer_function(k3vec, val, idir=idir, jdir=jdir):
                    kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                    kk[kk == 0] = 1
                    return k3vec[idir] * k3vec[jdir] * val / kk

                dij_k = base_field_mesh.apply(my_transfer_function,
                                              mode='complex',
                                              kind='wavenumber')
                del my_transfer_function
                dij_x = dij_k.compute(mode='real')
                if verbose:
                    rfield_print_info(dij_x, comm, 'd_%d%d: ' % (idir, jdir))

                # Add \sum_{i,j=0..2} d_ij(\vx)d_ij(\vx)
                #   = [d_00^2+d_11^2+d_22^2 + 2*(d_01^2+d_02^2+d_12^2)]
                if jdir == idir:
                    fac = 1.0
                else:
                    fac = 2.0
                out_rfield += fac * 1.5 * dij_x**2
                del dij_x, dij_k

    elif quadfield in ['shift', 'PsiNablaDelta']:
        # Get shift = \vPsi\cdot\nabla\delta
        for idir in range(3):
            # compute Psi_i
            def Psi_i_fcn(k3vec, val, idir=idir):
                kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                kk[kk == 0] = 1
                return -1.0j * k3vec[idir] * val / kk

            Psi_i_x = base_field_mesh.apply(
                Psi_i_fcn, mode='complex',
                kind='wavenumber').compute(mode='real')

            # compute nabla_i delta
            def grad_i_fcn(k3vec, val, idir=idir):
                return -1.0j * k3vec[idir] * val

            nabla_i_delta_x = base_field_mesh.apply(
                grad_i_fcn, mode='complex',
                kind='wavenumber').compute(mode='real')

            # multiply and add up in x space
            if idir == 0:
                out_rfield = Psi_i_x * nabla_i_delta_x
            else:
                out_rfield += Psi_i_x * nabla_i_delta_x

    elif quadfield in ['shift_two_meshs', 'PsiNablaDelta_two_meshs']:
        # Get shift = 0.5(\vPsi_1\cdot\nabla\delta_2 + \vPsi_2\cdot\nabla\delta_1)
        out_rfield = None
        for idir in range(3):
            
            def Psi_i_fcn(k3vec, val, idir=idir):
                kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                kk[kk == 0] = 1
                return -1.0j * k3vec[idir] * val / kk
            
            def grad_i_fcn(k3vec, val, idir=idir):
                return -1.0j * k3vec[idir] * val

            # compute Psi_i of mesh1
            Psi_i_x = base_field_mesh.apply(
                Psi_i_fcn, mode='complex',
                kind='wavenumber').compute(mode='real')

            # compute nabla_i delta of mesh2
            nabla_i_delta_x = second_base_field_mesh.apply(
                grad_i_fcn, mode='complex',
                kind='wavenumber').compute(mode='real')

            # multiply and add up in x space
            if out_rfield is None:
                out_rfield = 0.5 * Psi_i_x * nabla_i_delta_x
            else:
                out_rfield += 0.5 * Psi_i_x * nabla_i_delta_x
            del Psi_i_x, nabla_i_delta_x

            # also swap mesh1 and mesh2
            # compute Psi_i of mesh2
            Psi_i_x_prime = second_base_field_mesh.apply(
                Psi_i_fcn, mode='complex',
                kind='wavenumber').compute(mode='real')

            # compute nabla_i delta of mesh1
            nabla_i_delta_x_prime = base_field_mesh.apply(
                grad_i_fcn, mode='complex',
                kind='wavenumber').compute(mode='real')
            
            out_rfield += 0.5 * Psi_i_x_prime * nabla_i_delta_x_prime


    # elif quadfield in ['tidal_G2_par_LOS100', 'tidal_G2_par_LOS010', 'tidal_G2_par_LOS001']:
    #     if quadfield.endswith('100'):
    #         LOS_dir = 0
    #     elif quadfield.endswith('010'):
    #         LOS_dir = 1
    #     elif quadfield.endswith('001'):
    #         LOS_dir = 2
    #     else:
    #         raise Exception('invalid LOS')

    #     # compute nabla_parallel^2/nabla^2 G2
    #     G2_mesh = calc_quadratic_field(
    #         quadfield='tidal_G2',
    #         base_field_mesh=base_field_mesh,
    #         smoothing_of_base_field=smoothing_of_base_field,
    #         verbose=verbose)

    #     def my_parallel_transfer_function(k3vec, val, idir=LOS_dir, jdir=LOS_dir):
    #         kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
    #         kk[kk == 0] = 1
    #         # MS 25 Jul 2020: Include minus sign because (ik_2)^2=-k_2^2.
    #         return -k3vec[idir] * k3vec[jdir] * val / kk

    #     G2_parallel = G2_mesh.apply(my_parallel_transfer_function,
    #                                   mode='complex',
    #                                   kind='wavenumber')
    #     del my_parallel_transfer_function

    #     out_rfield = G2_parallel.compute(mode='real')
    #     del G2_parallel, G2_mesh



    elif quadfield == 'F2':
        # F2 = delta^2...
        out_rfield = calc_quadratic_field(
            quadfield='growth',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')
        # ... - shift
        out_rfield -= calc_quadratic_field(
            quadfield='shift',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')
        # ... + 2/7 tidal_G2
        out_rfield += 2. / 7. * calc_quadratic_field(
            quadfield='tidal_G2',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')

    elif quadfield == 'F2_two_meshs':
        # F2 = delta^2...
        out_rfield = calc_quadratic_field(
            quadfield='growth_two_meshs',
            base_field_mesh=base_field_mesh,
            second_base_field_mesh=second_base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')
        # ... - shift
        out_rfield -= calc_quadratic_field(
            quadfield='shift_two_meshs',
            base_field_mesh=base_field_mesh,
            second_base_field_mesh=second_base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')
        # ... + 2/7 tidal_G2
        out_rfield += 2. / 7. * calc_quadratic_field(
            quadfield='tidal_G2_two_meshs',
            base_field_mesh=base_field_mesh,
            second_base_field_mesh=second_base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')

    elif quadfield == 'velocity_G2':
        # G2 = delta^2...
        out_rfield = calc_quadratic_field(
            quadfield='growth',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')
        # ... - shift
        out_rfield -= calc_quadratic_field(
            quadfield='shift',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')
        # ... + 4/7 tidal_G2
        out_rfield += 4. / 7. * calc_quadratic_field(
            quadfield='tidal_G2',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')

    elif quadfield == 'velocity_G2_two_meshs':
        # G2 = delta^2...
        out_rfield = calc_quadratic_field(
            quadfield='growth_two_meshs',
            base_field_mesh=base_field_mesh,
            second_base_field_mesh=second_base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')
        # ... - shift
        out_rfield -= calc_quadratic_field(
            quadfield='shift_two_meshs',
            base_field_mesh=base_field_mesh,
            second_base_field_mesh=second_base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')
        # ... + 4/7 tidal_G2
        out_rfield += 4. / 7. * calc_quadratic_field(
            quadfield='tidal_G2_two_meshs',
            base_field_mesh=base_field_mesh,
            second_base_field_mesh=second_base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')

    elif quadfield in ['delta_G2', 'G2_delta']:
        # Get G2[delta] * delta
        out_rfield = (
            base_field_mesh.compute(mode='real')
            * calc_quadratic_field(
                quadfield='tidal_G2',
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose).compute(mode='real'))

        # take out the mean (already close to 0 but still subtract)
        mymean = out_rfield.cmean()
        if comm.rank == 0:
            print('Subtract mean of G2*delta: %g' % mymean)
        out_rfield -= mymean 


    elif quadfield in ['delta_G2_par_LOS100', 'delta_G2_par_LOS010', 'delta_G2_par_LOS001']:
        # Get delta * G2_parallel[delta]
        out_rfield = (
            base_field_mesh.compute(mode='real')
            * calc_quadratic_field(
                quadfield='tidal_G2_par_LOS%s' % (quadfield[-3:]),
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose).compute(mode='real'))

        # take out the mean (already close to 0 but still subtract)
        mymean = out_rfield.cmean()
        if comm.rank == 0:
            print('Subtract mean of G2par*delta: %g' % mymean)
        out_rfield -= mymean 


    elif quadfield in ['tidal_G3', 'tidal_G3_flipped_dij_sign']:
        # Get G3[delta]

        # Have 3/2 G2 delta = 3/2 (p1.p2)^2/(p1^2 p2^2) - 3/2
        # so
        # G3 = 3/2 G2 delta + delta^3 - (p1.p2)(p2.p3)(p1.p3)/(p1^2 p2^2 p3^2)

        # Compute 1 * delta^3(\vx)
        out_rfield = base_field_mesh.compute(mode='real')**3

        # Add 3/2 delta * G2[delta]
        out_rfield += (3./2. 
            * base_field_mesh.compute(mode='real')
            * calc_quadratic_field(
                quadfield='tidal_G2',
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose).compute(mode='real'))

        # Compute ppp = (p1.p2)(p2.p3)(p1.p3)/(p1^2 p2^2 p3^2)
        # = k.q k.p q.p / (k^2 q^2 p^2)
        # = k_i q_i k_j p_j q_l p_l / (k^2 q^2 p^2)
        # = sum_ijl d_ij(k) d_il(q) d_jl(p)
        # where we denoted p1=k, p2=q, p3=p.

        # Compute d_ij(x). It's symmetric in i<->j so only compute j>=i.
        # d_ij = k_ik_j/k^2*basefield(\vk).
        dij_x_dict = {}
        for idir in range(3):
            for jdir in range(idir, 3):

                def my_transfer_function(k3vec, val, idir=idir, jdir=jdir):
                    kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                    kk[kk == 0] = 1
                    return k3vec[idir] * k3vec[jdir] * val / kk

                dij_k = base_field_mesh.apply(my_transfer_function,
                                              mode='complex',
                                              kind='wavenumber')
                del my_transfer_function
                # do fft and convert field_mesh to RealField object
                dij_x = dij_k.compute(mode='real')
                del dij_k

                if verbose:
                    rfield_print_info(dij_x, comm, 'd_%d%d: ' % (idir, jdir))

                if quadfield == 'tidal_G3_flipped_dij_sign':
                    # flip sign of dij
                    dij_x *= -1.0

                dij_x_dict[(idir,jdir)] = dij_x
                del dij_x

        # get j<i by symmetry
        def get_dij_x(idir, jdir):
            if jdir>=idir:
                return dij_x_dict[(idir,jdir)]
            else:
                return dij_x_dict[(jdir,idir)]

        # Compute - sum_ijl d_ij(k) d_il(q) d_jl(p)
        for idir in range(3):
            for jdir in range(3):
                for ldir in range(3):
                    out_rfield -= (
                          get_dij_x(idir,jdir)
                        * get_dij_x(idir,ldir)
                        * get_dij_x(jdir,ldir) )

        # take out the mean (already close to 0 but still subtract)
        mymean = out_rfield.cmean()
        if comm.rank == 0:
            print('Subtract mean of G3: %g' % mymean)
        out_rfield -= mymean 

        if verbose:
            rfield_print_info(out_rfield, comm, 'G3: ')


    elif quadfield == 'Gamma3':
        # Get Gamma3[delta] = -4/7 * G2[G2[delta,delta],delta]

        tmp_G2 = calc_quadratic_field(
            quadfield='tidal_G2',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose)

        out_rfield = -4./7. * calc_quadratic_field(
            quadfield='tidal_G2_two_meshs',
            base_field_mesh=base_field_mesh,
            second_base_field_mesh=tmp_G2,
            smoothing_of_base_field=smoothing_of_base_field,
            smoothing_of_second_base_field=None,
            verbose=verbose).compute(mode='real')


    elif quadfield in [
        'delta1_par_LOS100', 'delta1_par_LOS010', 'delta1_par_LOS001', 
        'tidal_G2_par_LOS100', 'tidal_G2_par_LOS010', 'tidal_G2_par_LOS001', 
        'Gamma3_par_LOS100', 'Gamma3_par_LOS010', 'Gamma3_par_LOS001',
        'tidal_G3_par_LOS100', 'tidal_G3_par_LOS010', 'tidal_G3_par_LOS001', 
        'tidal_G3_flipped_dij_sign_par_LOS100', 'tidal_G3_flipped_dij_sign_par_LOS010', 'tidal_G3_flipped_dij_sign_par_LOS001'
    ]:
        if quadfield.endswith('LOS100'):
            LOS_dir = 0
        elif quadfield.endswith('LOS010'):
            LOS_dir = 1
        elif quadfield.endswith('LOS001'):
            LOS_dir = 2
        else:
            raise Exception('invalid LOS string in %s' % str(quadfield))


        if quadfield.startswith('delta1_par'):
            tmp_base_quadfield_str = 'delta1'
        elif quadfield.startswith('tidal_G2_par'):
            tmp_base_quadfield_str = 'tidal_G2'
        elif quadfield.startswith('Gamma3_par'):
            tmp_base_quadfield_str = 'Gamma3'
        elif quadfield.startswith('tidal_G3_par'):
            tmp_base_quadfield_str = 'tidal_G3'
        elif quadfield.startswith('tidal_G3_flipped_dij_sign_par'):
            tmp_base_quadfield_str = 'tidal_G3_flipped_dij_sign'
        else:
            raise Exception('Invalid quadfield %s' % str(quadfield))

        # compute nabla_parallel^2/nabla^2 tmp_base_quadfield
        if tmp_base_quadfield_str == 'delta1':
            tmp_base_quadfield_mesh = FieldMesh(base_field_mesh.compute(
                mode='real'))
        else:
            tmp_base_quadfield_mesh = calc_quadratic_field(
                quadfield=tmp_base_quadfield_str,
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose)

        def my_parallel_transfer_function(k3vec, val, idir=LOS_dir, jdir=LOS_dir):
            kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
            kk[kk == 0] = 1
            # MS 25 Jul 2020: Include minus sign because (ik_2)^2=-k_2^2.
            # MS 29 Jul 2020: Drop minus sign because 1/nabla^2 gives another minus sign.
            return k3vec[idir] * k3vec[jdir] * val / kk

        tmp_base_quadfield_parallel = tmp_base_quadfield_mesh.apply(
            my_parallel_transfer_function,
            mode='complex',
            kind='wavenumber')
        del my_parallel_transfer_function

        out_rfield = tmp_base_quadfield_parallel.compute(mode='real')
        del tmp_base_quadfield_parallel, tmp_base_quadfield_mesh


    elif quadfield in ['velocity_G2_par_LOS001_two_meshs']:
        LOS_dir = 2
        tmp_base_quadfield_str = 'velocity_G2_two_meshs'

        # compute nabla_parallel^2/nabla^2 tmp_base_quadfield
        tmp_base_quadfield_mesh = calc_quadratic_field(
            quadfield=tmp_base_quadfield_str,
            base_field_mesh=base_field_mesh,
            second_base_field_mesh=second_base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose)

        def my_parallel_transfer_function(k3vec, val, idir=LOS_dir, jdir=LOS_dir):
            kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
            kk[kk == 0] = 1
            # MS 29 Jul 2020: No minus sign because -(ik_2)^2/k^2=k_2^2/k^2.
            return k3vec[idir] * k3vec[jdir] * val / kk

        tmp_base_quadfield_parallel = tmp_base_quadfield_mesh.apply(
            my_parallel_transfer_function,
            mode='complex',
            kind='wavenumber')
        del my_parallel_transfer_function

        out_rfield = tmp_base_quadfield_parallel.compute(mode='real')
        del tmp_base_quadfield_parallel, tmp_base_quadfield_mesh

    elif quadfield == 'delta1_par_G2_par_LOS001':
        out_rfield = (
            calc_quadratic_field(
                quadfield='delta1_par_LOS001',
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose).compute(mode='real')
            * calc_quadratic_field(
                quadfield='tidal_G2_par_LOS001',
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose).compute(mode='real') )


    elif quadfield in ['RSDLPT_Q2_LOS001']:
        # Q2 = sum_i d_i2 d_i2
        # d_ij = -k_ik_j/k^2*basefield(\vk).

        out_rfield = None
        jdir = 2
        for idir in range(3):

            def my_transfer_function_dij(k3vec, val, idir=idir, jdir=jdir):
                kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                kk[kk == 0] = 1
                return k3vec[idir] * k3vec[jdir] * val / kk

            dij_k = base_field_mesh.apply(my_transfer_function_dij,
                                          mode='complex',
                                          kind='wavenumber')
            del my_transfer_function_dij
            # do fft and convert field_mesh to RealField object
            dij_x = dij_k.compute(mode='real')
            del dij_k

            if verbose:
                rfield_print_info(dij_x, comm, 'd_%d%d: ' % (idir, jdir))

            if out_rfield is None:
                out_rfield = dij_x**2
            else:
                out_rfield += dij_x**2


    elif quadfield in ['RSDLPT_K3_LOS001']:
        # K3 = sum_i d_i2 dG2_i2
        # where d_ij = k_ik_j/k^2*basefield(\vk)
        # and dG2_ij = k_ik_j/k^2*G2(\vk).
        tmp_G2_mesh = calc_quadratic_field(
            quadfield='tidal_G2',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose)

        out_rfield = None
        jdir = 2
        for idir in range(3):

            def my_transfer_function_dij(k3vec, val, idir=idir, jdir=jdir):
                kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                kk[kk == 0] = 1
                return k3vec[idir] * k3vec[jdir] * val / kk

            dij_x = base_field_mesh.apply(my_transfer_function_dij,
                                          mode='complex',
                                          kind='wavenumber').compute(mode='real')

            dijG2_x = tmp_G2_mesh.apply(my_transfer_function_dij,
                                          mode='complex',
                                          kind='wavenumber').compute(mode='real')

            del my_transfer_function_dij

            if out_rfield is None:
                out_rfield = dij_x * dijG2_x
            else:
                out_rfield += dij_x * dijG2_x


    elif quadfield in ['RSDLPT_S3Ia','RSDLPT_S3IIa_LOS001']:
        # S3Ia = sum_i d_i where d_i = (k_i/k^2 G2) (k_i delta_1)
        # (or delta_1 -> delta_1 parallel for S3IIa)
        tmp_G2_mesh = calc_quadratic_field(
            quadfield='tidal_G2',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose)

        if quadfield == 'RSDLPT_S3Ia':
            tmp_delta1_mesh = FieldMesh(base_field_mesh.compute(mode='real'))
        elif quadfield == 'RSDLPT_S3IIa_LOS001':
            tmp_delta1_mesh =  calc_quadratic_field(
                quadfield='delta1_par_LOS001',
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose)

        out_rfield = None
        for idir in range(3):

            def my_trf_for_G2(k3vec, val, idir=idir):
                kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                kk[kk == 0] = 1
                return k3vec[idir] * val / kk

            def my_trf_for_delta1(k3vec, val, idir=idir):
                return k3vec[idir] * val

            G2_filtered = tmp_G2_mesh.apply(my_trf_for_G2,
                                          mode='complex',
                                          kind='wavenumber').compute(mode='real')

            delta1_filtered = tmp_delta1_mesh.apply(my_trf_for_delta1,
                                          mode='complex',
                                          kind='wavenumber').compute(mode='real')

            del my_trf_for_G2, my_trf_for_delta1

            if out_rfield is None:
                out_rfield = G2_filtered * delta1_filtered
            else:
                out_rfield += G2_filtered * delta1_filtered


    elif quadfield in ['RSDLPT_S3Ib_LOS001','RSDLPT_S3IIb_LOS001']:
        # S3Ia = (k_2/k^2 G2) (k_2 delta_1)

        tmp_G2_mesh = calc_quadratic_field(
            quadfield='tidal_G2',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose)

        if quadfield == 'RSDLPT_S3Ib_LOS001':
            tmp_delta1_mesh = FieldMesh(base_field_mesh.compute(mode='real'))
        elif quadfield == 'RSDLPT_S3IIb_LOS001':
            tmp_delta1_mesh =  calc_quadratic_field(
                quadfield='delta1_par_LOS001',
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose)

        idir = 2

        def my_trf_for_G2(k3vec, val, idir=idir):
            kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
            kk[kk == 0] = 1
            return k3vec[idir] * val / kk

        def my_trf_for_delta1(k3vec, val, idir=idir):
            return k3vec[idir] * val

        G2_filtered = tmp_G2_mesh.apply(my_trf_for_G2,
                                      mode='complex',
                                      kind='wavenumber').compute(mode='real')

        delta1_filtered = tmp_delta1_mesh.apply(my_trf_for_delta1,
                                      mode='complex',
                                      kind='wavenumber').compute(mode='real')

        del my_trf_for_G2, my_trf_for_delta1

        out_rfield = G2_filtered * delta1_filtered


    elif quadfield in ['RSDLPT_calQ2_LOS001']:
        # calQ2 = Q2 - delta1*delta1_parallel
        out_rfield = calc_quadratic_field(
            quadfield='RSDLPT_Q2_LOS001',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose)

        out_rfield -= (
            base_field_mesh.compute(mode='real')
            * calc_quadratic_field(
            quadfield='delta1_par_LOS001',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose))

    elif quadfield in ['RSDLPT_calQ3_LOS001']:
        # calQ3 = Q3 - delta1 Q2 - 1/2 delta1_par G2
        out_rfield = calc_quadratic_field(
            quadfield='RSDLPT_Q3_LOS001',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose)

        out_rfield -= (
            base_field_mesh.compute(mode='real')
            * calc_quadratic_field(
            quadfield='RSDLPT_Q2_LOS001',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose))

        out_rfield -= 0.5 * (
            calc_quadratic_field(
                quadfield='delta1_par_LOS001',
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose)
            * calc_quadratic_field(
                quadfield='tidal_G2',
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose)
            )


    elif quadfield in ['RSDLPT_Q3_LOS001']:
        # Q3 = sum_m,n d_m2 d_mn d_n2
        # d_ij = k_ik_j/k^2*basefield(\vk).
        # Compute d_ij(x). It's symmetric in i<->j so only compute j>=i.
        dij_x_dict = {}
        for idir in range(3):
            for jdir in range(idir, 3):

                def my_dij_transfer_function(k3vec, val, idir=idir, jdir=jdir):
                    kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                    kk[kk == 0] = 1
                    return k3vec[idir] * k3vec[jdir] * val / kk

                dij_k = base_field_mesh.apply(my_dij_transfer_function,
                                              mode='complex',
                                              kind='wavenumber')
                del my_dij_transfer_function
                # do fft and convert field_mesh to RealField object
                dij_x = dij_k.compute(mode='real')
                del dij_k

                if verbose:
                    rfield_print_info(dij_x, comm, 'd_%d%d: ' % (idir, jdir))

                dij_x_dict[(idir,jdir)] = dij_x
                del dij_x

        # get j<i by symmetry
        def get_dij_x(idir, jdir):
            if jdir>=idir:
                return dij_x_dict[(idir,jdir)]
            else:
                return dij_x_dict[(jdir,idir)]

        # Compute sum_nm d_2m(k) d_mn(q) d_n2(p), assume LOS=z direction.
        out_rfield = 0.0 * base_field_mesh.compute(mode='real')
        for mdir in range(3):
            for ndir in range(3):
                out_rfield += (
                      get_dij_x(2,mdir)
                    * get_dij_x(mdir,ndir)
                    * get_dij_x(ndir,2) )

        # take out the mean (already close to 0 but still subtract)
        mymean = out_rfield.cmean()
        if comm.rank == 0:
            print('Subtract mean of %s: %g' % (quadfield, mymean))
        out_rfield -= mymean




    elif quadfield.startswith('PsiDot1_'):
        # get PsiDot \equiv \sum_n n*Psi^{(n)} up to 1st order
        assert quadfield in ['PsiDot1_0', 'PsiDot1_1', 'PsiDot1_2']
        component = int(quadfield[-1])
        out_rfield = get_displacement_from_density_rfield(
            base_field_mesh.compute(mode='real'),
            component=component,
            Psi_type='Zeldovich',
            smoothing=smoothing_of_base_field,
            RSD=False
            )

    elif quadfield.startswith('PsiDot2_'):
        # get PsiDot \equiv \sum_n n*Psi^{(n)} up to 2nd order
        assert quadfield in ['PsiDot2_0', 'PsiDot2_1', 'PsiDot2_2']
        component = int(quadfield[-1])

        # PsiDot \equiv \sum_n n*Psi^{(n)}
        out_rfield = get_displacement_from_density_rfield(
            base_field_mesh.compute(mode='real'),
            component=component,
            Psi_type='2LPT',
            prefac_Psi_1storder=1.0,
            prefac_Psi_2ndorder=2.0,  # include n=2 factor
            smoothing=smoothing_of_base_field,
            RSD=False
            )

    else:
        raise Exception("quadfield %s not implemented" % str(quadfield))

    return FieldMesh(out_rfield)


def linear_rescale_fac(current_scale_factor,
                       desired_scale_factor,
                       cosmo_params=None):
    if desired_scale_factor is None or current_scale_factor is None:
        raise Exception("scale factors must be not None")
    if desired_scale_factor > 1.0 or current_scale_factor > 1.0:
        raise Exception("scale factors must be <=1")

    if desired_scale_factor == current_scale_factor:
        rescalefac = 1.0
    else:
        # Factor to linearly rescale delta to desired redshift
        assert (cosmo_params is not None)
        cosmo = CosmoModel(**cosmo_params)
        calc_Da = generate_calc_Da(cosmo=cosmo, verbose=False)
        rescalefac = calc_Da(desired_scale_factor) / calc_Da(
            current_scale_factor)
        #del cosmo
    return rescalefac

def catalog_persist(cat, columns=None):
    """
    Return a CatalogSource, where the selected columns are
    computed and persist in memory.
    """

    import dask.array as da
    if columns is None:
        columns = cat.columns

    r = {}
    for key in columns:
        if key in cat.columns:
            r[key] = cat[key]

    r = da.compute(r)[0] # particularity of dask

    from nbodykit.source.catalog.array import ArrayCatalog
    c = ArrayCatalog(r, comm=cat.comm)
    c.attrs.update(cat.attrs)

    return c


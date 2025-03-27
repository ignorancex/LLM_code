import numpy as np
import sys
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay

class BispTreeLevel:
    """
    An exemplary module to compute tree-level model predictions for the galaxy bispectrum in real space.
    """

    def __init__(self, fname_Plinear=None, Plinear=None):
        """
        Initialise BispTreeLevel object. Takes either a filename storing the linear power spectrum (two columns: k, P_lin), or an array [k, P_lin].

        Args:
            fname_Plinear (str): filename storing linear power spectrum
            Plinear (float): numpy array with two columns corresponding to k and P_lin(k)
        """
        self.Pspline = None
        self.neff1 = 0.
        self.neff2 = 0.
        self.kmin = 0.
        self.kmax = 0.
        self.PLmin = 0.
        self.PLmax = 0.
        if fname_Plinear:
            self.init_Plinear_from_file(fname_Plinear)
        if Plinear:
            self.init_Plinear(Plinear)

    def init_Plinear(self, Plinear):
        """
        Generate spline of linear power spectrum from array.

        Args:
            Plinear (float): numpy array with two columns corresponding to k and P_lin(k)
        """
        self.Pspline = interp1d(Plinear[:,0],Plinear[:,1],kind='cubic')
        self.compute_neff(Plinear)

    def init_Plinear_from_file(self, fname_Plinear):
        """
        Generate spline of linear power spectrum from filename.

        Args:
            fname_Plinear (str): filename storing linear power spectrum in the format k, P_lin(k)
        """
        Plinear = np.loadtxt(fname_Plinear)
        self.Pspline = interp1d(Plinear[:,0],Plinear[:,1],kind='cubic')
        self.compute_neff(Plinear)

    def compute_neff(self, Plinear):
        """
        Compute effective spectral indices in the low-k and high-k limit of the input linear power spectrum.
        """
        dlp = np.log10(Plinear[2,1]) - np.log10(Plinear[0,1])
        dlk = np.log10(Plinear[2,0]) - np.log10(Plinear[0,0])
        self.neff1 = dlp/dlk
        dlp = np.log10(Plinear[-1,1]) - np.log10(Plinear[-3,1])
        dlk = np.log10(Plinear[-1,0]) - np.log10(Plinear[-3,0])
        self.neff2 = dlp/dlk
        self.kmin = Plinear[0,0]
        self.kmax = Plinear[-1,0]
        self.PLmin = Plinear[0,1]
        self.PLmax = Plinear[-1,1]

    def PL(self, k):
        """
        Return splined linear power spectrum, extended below kmin and beyond kmax of the input power spectrum using the effective spectral indices.

        Args:
            k (float): single value or list/array of scales

        Returns:
            Linear power spectrum at scale(s) k
        """
        if not isinstance(k,list) and not isinstance(k,np.ndarray):
            k = np.array([k])
        P = np.zeros(k.shape[0])
        for i,kk in enumerate(k):
            if kk >= self.kmin and kk <= self.kmax:
                P[i] = self.Pspline(kk)
            elif kk < self.kmin:
                P[i] = self.PLmin*(kk/self.kmin)**self.neff1
            elif kk > self.kmax:
                P[i] = self.PLmax*(kk/self.kmax)**self.neff2
        return P

    def F2(self, k1, k2, k3):
        """
        Return value of second-order SPT kernel F2[vec(k1),vec(k2)].

        Args:
            k1, k2 (float): magnitudes of vectors vec(k1) and vec(k2)
            k3 (float): magnitude of vec(k1)+vec(k2)

        Returns:
            Value of F2
        """
        mu = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
        return 5./7 + mu/2*(k1/k2 + k2/k1) + 2./7*mu**2

    def K(self, k1, k2, k3):
        """
        Return value of second-order Galileon operator kernel K[vec(k1),vec(k2)].

        Args:
            k1, k2 (float): magnitudes of vectors vec(k1) and vec(k2)
            k3 (float): magnitude of vec(k1)+vec(k2)

        Returns:
            Value of K
        """
        mu = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
        return mu**2 - 1.

    def B211(self, k):
        """
        Return tree-level galaxy bispectrum contributions.

        Args:
            k (float): list/array of three columns storing the three triangle sides for a set of configurations

        Returns:
            List of bispectrum contributions proportional to b1^3, b1^2*b2, b1^2*gamma2 for the provided list of configurations.
        """
        if not isinstance(k,np.ndarray):
            k = np.array([k])
        elif isinstance(k,np.ndarray) and k.ndim == 1:
            k = k.reshape(1,3)
        B = np.zeros([k.shape[0],k.shape[1]])
        for n in range(k.shape[0]):
            k1,k2,k3 = k[n]
            P1 = self.PL(k1)
            P2 = self.PL(k2)
            P3 = self.PL(k3)
            B[n,0] = 2*(self.F2(k1,k2,k3)*P1*P2 + self.F2(k2,k3,k1)*P2*P3 + self.F2(k3,k1,k2)*P3*P1)
            B[n,1] = P1*P2 + P2*P3 + P3*P1
            B[n,2] = 2*(self.K(k1,k2,k3)*P1*P2 + self.K(k2,k3,k1)*P2*P3 + self.K(k3,k1,k2)*P3*P1)
        return B


class BispDelaunay:
    """
    This module corrects bispectrum model predictions for the binning effect using (linear) Delaunay interpolation.
    The BispDelaunay object is initialised with the scale cuts of the measurements (kmin and kmax) and their bin width.
    These boundary values are used to define all allowed triangle configuration (excluding open triangles for the moment),
    which are assumed to be identical with the measured configurations ("target configurations"). Each configuration is
    decomposed into a set of tetrahedra that define the interpolation grid (resolution can be controlled via the function
    "set_refinement_settings") and the Delaunay binning matrix is computed using the function "compute_Delaunay_matrix".
    Further details are described in the accompanying Jupyter notebook.
    """

    def __init__(self, kmin, kmax, dk, kf):
        """
        Initialise BispDelaunay object.

        Args:
            kmin (float): minimum scale of the measurements at the bin centre (in units of kf)
            kmax (float): maximum scale of the measurements at the bin centre (in units of kf)
            dk (float): bin width (in units of kf)
            kf (float): fundamental frequency
        """
        self.kmin = kmin
        self.kmin_ph = kmin*kf
        self.kmax = kmax
        self.kmax_ph = kmax*kf
        self.dk = dk
        self.dk_ph = dk*kf
        self.kf = kf
        self.tri = None
        self.tri = self.get_triangle_conf()

        self.nstep = [2,3,5]
        self.squeezed_cut = 0.05
        self.collinear_cut = 0.8
        self.midpoint = True

        self.vertices = None
        self.unique_vertices = None
        self.delaunay_grids = None
        self.vertex_lookup = None
        self.alpha_vertices = None
        self.DM_v = None
        self.Btable_vertices_unique = None

    def get_triangle_conf(self):
        """
        Return unique triangle configurations as multiples of the bin width, ordered as k1 >= k2 >= k3.
        """
        try:
            self.tri
        except NameError:
            self.tri = None
        if self.tri is not None:
            return self.tri
        else:
            tri = []
            for i1 in range(int(self.kmin*1./self.dk),int(self.kmax*1./self.dk)+1):
                for i2 in range(int(self.kmin*1./self.dk),i1+1):
                    for i3 in range(int(self.kmin*1./self.dk),i2+1):
                        if i1 <= i2+i3:
                            tri.append([i1*self.dk,i2*self.dk,i3*self.dk])
            self.Ntri = len(tri)
            return np.array(tri)

    def set_refinement_settings(self, nstep=[2,3,5], squeezed_cut=0.05, collinear_cut=0.8, midpoint=True):
        """
        Set settings that are used to define the vertices of the tetrahedra.

        Args:
            nstep (list of three ints): refinement levels for generic configs, configs with non-cubical integration volume, squeezed/collinear configs
            squeezed_cut (float): squeezed configs = k3/(k1+k2) < squeezed_cut
            collinear_cut (float): collinear configs = (k2+k3)/k1 < collinear_cut
            midpoint (bool): if true, use bin centre as additional vertex for generic configs
        """
        self.nstep = nstep
        self.squeezed_cut = squeezed_cut
        self.collinear_cut = collinear_cut
        self.midpoint = midpoint

    def generate_vertices(self):
        """
        Generate grid of vertices according to refinement settings.
        """
        self.vertices = []
        self.refinement = np.zeros(self.tri.shape[0],dtype=int)
        for n,tri in enumerate(self.tri):
            limits = []
            k1, k2, k3 = tri
            if np.abs(k1-k2-k3) <= self.dk or np.any(tri == self.dk): # configurations with non-cubical integration volume
                nstep = self.nstep[1]
            else: # cubic integration volume
                nstep = self.nstep[0]
            if k3*1./(k1+k2) < self.squeezed_cut: # highly squeezed configurations
                nstep = self.nstep[2]
            if (k2+k3)*1./k1 <= self.collinear_cut: # collinear (or close to coll.) configurations
                nstep = self.nstep[2]
            self.refinement[n] = nstep
            for x1 in range(nstep):
                for x2 in range(nstep):
                    q1 = k1 - self.dk*1./2 + self.dk*1./(nstep-1)*x1
                    q2 = k2 - self.dk*1./2 + self.dk*1./(nstep-1)*x2
                    q3min = max(k3-self.dk*1./2,abs(q1-q2))
                    q3max = min(k3+self.dk*1./2,q1+q2)
                    if abs(q3min-q3max) < 1e-4:
                        limits.append([q1,q2,q3min])
                    elif q3min < q3max:
                        limits.append([q1,q2,q3min])
                        limits.append([q1,q2,q3max])
                        if np.abs(q3max-q3min-self.dk) < 1e-4:
                            for i in range(nstep-2):
                                limits.append([q1,q2,q3min+(q3max-q3min)*1./(nstep-1)*(i+1)])
                        elif np.abs(q3max-q3min-self.dk*1./2) < 1e-4:
                            for i in range(nstep-4):
                                limits.append([q1,q2,q3min+(q3max-q3min)*1./(nstep-3)*(i+1)])
                    if nstep == 2 and self.midpoint:
                        limits.append([k1,k2,k3])
            self.vertices.append(np.unique(np.array(limits),axis=0))

    def generate_Delaunay_triangularisation(self):
        """
        Generate Delaunay triangularisation for each bispectrum triangle configuration (k1,k2,k3).
        """
        if self.vertices is None:
            self.generate_vertices()
        self.delaunay_grids = []
        for n in range(self.Ntri):
            self.delaunay_grids.append(Delaunay(self.vertices[n]))
        self.generate_unique_vertices()

    def generate_unique_vertices(self):
        """
        Generate list of unique vertices by removing duplicates from different bispectrum triangle configurations.
        """
        self.unique_vertices = np.array([])
        for n in range(self.Ntri):
            self.unique_vertices = np.vstack([self.unique_vertices,self.delaunay_grids[n].points]) if self.unique_vertices.size else self.delaunay_grids[n].points
        self.unique_vertices = np.unique(self.unique_vertices,axis=0)

    def generate_vertex_lookup(self):
        """
        Generate map between list of unique vertices and the vertices of a given Delaunay triangularisation.
        """
        if self.unique_vertices is None:
            self.generate_unique_vertices()
        self.vertex_lookup = []
        for n in range(self.Ntri):
            lookup = -np.ones([self.delaunay_grids[n].simplices.shape[0],4],dtype=int)
            for i,simp in enumerate(self.delaunay_grids[n].simplices):
                for j,p in enumerate(self.delaunay_grids[n].points[simp]):
                    lookup[i,j] = np.where((self.unique_vertices[:,0] == p[0]) & (self.unique_vertices[:,1] == p[1]) & (self.unique_vertices[:,2] == p[2]))[0]
            self.vertex_lookup.append(lookup)

    def collect_edges_from_simplex(self, simplex):
        edges = set()

        def sorted_tuple(a,b):
            return (a,b) if a < b else (b,a)
        # Add edges of tetrahedron (sorted so we don't add an edge twice, even if it comes in reverse order).
        i0 = simplex[0]
        i1 = simplex[1]
        i2 = simplex[2]
        i3 = simplex[3]
        edges.add(sorted_tuple(i0,i1))
        edges.add(sorted_tuple(i0,i2))
        edges.add(sorted_tuple(i0,i3))
        edges.add(sorted_tuple(i1,i2))
        edges.add(sorted_tuple(i1,i3))
        edges.add(sorted_tuple(i2,i3))
        return edges

    def plot_tets(self, ax, simp, tet, color='C2'):
        for s in simp:
            edges = self.collect_edges_from_simplex(tet.simplices[s])
            x = np.array([])
            y = np.array([])
            z = np.array([])
            for (i,j) in edges:
                x = np.append(x, [tet.points[i, 0], tet.points[j, 0], np.nan])
                y = np.append(y, [tet.points[i, 1], tet.points[j, 1], np.nan])
                z = np.append(z, [tet.points[i, 2], tet.points[j, 2], np.nan])
            ax.plot3D(x, y, z, color=color, lw=0.5)

    def plot_simplices(self, tri_indices, ax, color='C2'):
        """
        Plot all tetrahedra (simplices) corresponding to a given set of triangle configurations.

        Args:
            tri_indices (int): single value or list/array of triangle indices corresponding to the desired triangle configurations
            ax: a matplotlib axes object
            color: a matplotlib identifier for the color of the tetrahedra edges
        """
        if not isinstance(tri_indices,list) and not isinstance(tri_indices,np.ndarray):
            tri_indices = [tri_indices]
        for tri_index in tri_indices:
            self.plot_tets(ax, np.arange(self.delaunay_grids[tri_index].simplices.shape[0]), self.delaunay_grids[tri_index], color=color)

    def alpha(self, n, v):
        """
        Interpolation kernels from integration over barycentric coordinates (lambda_1, lambda_2, lambda_3).

        Args:
            n (int): integrated with additional factor of lambda_n (n=1..4) or without (n=0)
            v (4x3 array of ints): coordinates of the four tetrahedron vertices

        Returns:
            Value of interpolation kernel for given tetrahedron.
        """
        u2 = v[1,0]
        u3 = v[1,1]
        u1 = v[0,2]
        u5 = v[1,2]
        u7 = v[2,0]
        u10 = v[2,1]
        u15 = v[2,2]
        u20 = v[3,0]
        u28 = v[3,1]
        u38 = v[3,2]
        u60 = v[0,1]
        u63 = 2.e0*u38

        if n == 0:
            return 1.388888889e-3*(u1*u10*u2 + 2.e0*u10*u15*u2 + u1*u10*u20 + 2.e0 \
                    *u10*u15*u20 + u1*u2*u28 + u15*u2*u28 + 2.e0*u1*u20*u28 + 2.e0*u15 \
                    *u20*u28 + 2.e0*u1*u2*u3 + 2.e0*u15*u2*u3 + u1*u20*u3 + u15*u20*u3 \
                    + u10*u2*u38 + 2.e0*u10*u20*u38 + 2.e0*u2*u28*u38 + 6.e0*u20*u28*u38 \
                    + 2.e0*u2*u3*u38 + 2.e0*u20*u3*u38 + 2.e0*u10*u2*u5 + u10*u20*u5 \
                    + 2.e0*u2*u28*u5 + 2.e0*u20*u28*u5 + 6.e0*u2*u3*u5 + 2.e0*u20*u3*u5 \
                    + 2.e0*u1*u10*u7 + 6.e0*u10*u15*u7 + u1*u28*u7 + 2.e0*u15*u28*u7 \
                    + u1*u3*u7 + 2.e0*u15*u3*u7 + 2.e0*u10*u38*u7 + 2.e0*u28*u38*u7 \
                    + u3*u38*u7 + 2.e0*u10*u5*u7 + u28*u5*u7 + 2.e0*u3*u5*u7 + u60*(u15 \
                    *u20 + 2.e0*u20*u38 + u20*u5 + u2*(u15 + u38 + 2.e0*u5) + 2.e0*u15*u7 \
                    + u38*u7 + u5*u7 + 2.e0*u1*(u2 + u20 + u7)) + (2.e0*u10*u15 + u15*u28 \
                    + u15*u3 + 2.e0*u1*(u10 + u28 + u3) + u10*u38 + 2.e0*u28*u38 + u3*u38 \
                    + u10*u5 + u28*u5 + 2.e0*u3*u5 + 2.e0*(3.e0*u1 + u15 + u38 + u5)*u60)*v[0,0])
        elif n == 1:
            return 1.984126984e-4*(2.e0*u1*u10*u2 + 2.e0*u10*u15*u2 + 2.e0*u1 \
                    *u10*u20 + 2.e0*u10*u15*u20 + 2.e0*u1*u2*u28 + u15*u2*u28 \
                    + 4.e0*u1*u20*u28 + 2.e0*u15*u20*u28 + 4.e0*u1*u2*u3 + 2.e0 \
                    *u15*u2*u3 + 2.e0*u1*u20*u3 + u15*u20*u3 + u10*u2*u38 + 2.e0 \
                    *u10*u20*u38 + 2.e0*u2*u28*u38 + 6.e0*u20*u28*u38 + 2.e0*u2*u3 \
                    *u38 + 2.e0*u20*u3*u38 + 2.e0*u10*u2*u5 + u10*u20*u5 + 2.e0*u2 \
                    *u28*u5 + 2.e0*u20*u28*u5 + 6.e0*u2*u3*u5 + 2.e0*u20*u3*u5 \
                    + 4.e0*u1*u10*u7 + 6.e0*u10*u15*u7 + 2.e0*u1*u28*u7 + 2.e0*u15 \
                    *u28*u7 + 2.e0*u1*u3*u7 + 2.e0*u15*u3*u7 + 2.e0*u10*u38*u7 + 2.e0 \
                    *u28*u38*u7 + u3*u38*u7 + 2.e0*u10*u5*u7 + u28*u5*u7 + 2.e0*u3*u5 \
                    *u7 + 2.e0*u60*(u15*u20 + 2.e0*u20*u38 + u20*u5 + u2*(u15 + u38 \
                    + 2.e0*u5) + 2.e0*u15*u7 + u38*u7 + u5*u7 + 3.e0*u1*(u2 + u20 + u7)) \
                    + 2.e0*(2.e0*u10*u15 + u15*u28 + u15*u3 + 3.e0*u1*(u10 + u28 + u3) \
                    + u10*u38 + 2.e0*u28*u38 + u3*u38 + u10*u5 + u28*u5 + 2.e0*u3*u5 \
                    + 3.e0*(4.e0*u1 + u15 + u38 + u5)*u60)*v[0,0])
        elif n == 2:
            return 1.984126984e-4*(2.e0*u1*u10*u2 + 4.e0*u10*u15*u2 + u1*u10*u20 \
                    + 2.e0*u10*u15*u20 + 2.e0*u1*u2*u28 + 2.e0*u15*u2*u28 + 2.e0*u1 \
                    *u20*u28 + 2.e0*u15*u20*u28 + 6.e0*u1*u2*u3 + 6.e0*u15*u2*u3 \
                    + 2.e0*u1*u20*u3 + 2.e0*u15*u20*u3 + 2.e0*u10*u2*u38 + 2.e0*u10 \
                    *u20*u38 + 4.e0*u2*u28*u38 + 6.e0*u20*u28*u38 + 6.e0*u2*u3*u38 \
                    + 4.e0*u20*u3*u38 + 6.e0*u10*u2*u5 + 2.e0*u10*u20*u5 + 6.e0*u2 \
                    *u28*u5 + 4.e0*u20*u28*u5 + 2.4e1*u2*u3*u5 + 6.e0*u20*u3*u5 + 2.e0 \
                    *u1*u10*u7 + 6.e0*u10*u15*u7 + u1*u28*u7 + 2.e0*u15*u28*u7 + 2.e0 \
                    *u1*u3*u7 + 4.e0*u15*u3*u7 + 2.e0*u10*u38*u7 + 2.e0*u28*u38*u7 \
                    + 2.e0*u3*u38*u7 + 4.e0*u10*u5*u7 + 2.e0*u28*u5*u7 + 6.e0*u3*u5*u7 \
                    + u60*(u15*u20 + 2.e0*u20*u38 + 2.e0*u20*u5 + 2.e0*u2*(u15 + u38 \
                    + 3.e0*u5) + 2.e0*u15*u7 + u38*u7 + 2.e0*u5*u7 + 2.e0*u1*(2.e0*u2 \
                    + u20 + u7)) + (2.e0*u10*u15 + u15*u28 + 2.e0*u15*u3 + 2.e0*u1*(u10 \
                    + u28 + 2.e0*u3) + u10*u38 + 2.e0*u28*u38 + 2.e0*u3*u38 + 2.e0*u10*u5 \
                    + 2.e0*u28*u5 + 6.e0*u3*u5 + 2.e0*(3.e0*u1 + u15 + u38 + 2.e0*u5)*u60)*v[0,0])
        elif n == 3:
            return 1.984126984e-4*(2.e0*u1*u10*u2 + 6.e0*u10*u15*u2 + 2.e0*u1 \
                    *u10*u20 + 6.e0*u10*u15*u20 + u1*u2*u28 + 2.e0*u15*u2*u28 \
                    + 2.e0*u1*u20*u28 + 4.e0*u15*u20*u28 + 2.e0*u1*u2*u3 + 4.e0*u15 \
                    *u2*u3 + u1*u20*u3 + 2.e0*u15*u20*u3 + 2.e0*u10*u2*u38 + 4.e0*u10 \
                    *u20*u38 + 2.e0*u2*u28*u38 + 6.e0*u20*u28*u38 + 2.e0*u2*u3*u38 \
                    + 2.e0*u20*u3*u38 + 4.e0*u10*u2*u5 + 2.e0*u10*u20*u5 + 2.e0*u2 \
                    *u28*u5 + 2.e0*u20*u28*u5 + 6.e0*u2*u3*u5 + 2.e0*u20*u3*u5 + 6.e0 \
                    *u1*u10*u7 + 2.4e1*u10*u15*u7 + 2.e0*u1*u28*u7 + 6.e0*u15*u28*u7 \
                    + 2.e0*u1*u3*u7 + 6.e0*u15*u3*u7 + 6.e0*u10*u38*u7 + 4.e0*u28*u38 \
                    *u7 + 2.e0*u3*u38*u7 + 6.e0*u10*u5*u7 + 2.e0*u28*u5*u7 + 4.e0*u3 \
                    *u5*u7 + u60*(2.e0*u15*u20 + 2.e0*u20*u38 + u20*u5 + u2*(u38 \
                    + 2.e0*u5 + u63) + 6.e0*u15*u7 + 2.e0*u38*u7 + 2.e0*u5*u7 + 2.e0 \
                    *u1*(u2 + u20 + 2.e0*u7)) + (6.e0*u10*u15 + 2.e0*u15*u28 + 2.e0 \
                    *u15*u3 + 2.e0*u1*(2.e0*u10 + u28 + u3) + 2.e0*u10*u38 + 2.e0*u28 \
                    *u38 + u3*u38 + 2.e0*u10*u5 + u28*u5 + 2.e0*u3*u5 + 2.e0*u60*(3.e0*u1 \
                    + u38 + u5 + u63))*v[0,0])
        elif n == 4:
            return 1.984126984e-4*(u1*u10*u2 + 2.e0*u10*u15*u2 + 2.e0*u1*u10*u20 \
                    + 4.e0*u10*u15*u20 + 2.e0*u1*u2*u28 + 2.e0*u15*u2*u28 + 6.e0*u1 \
                    *u20*u28 + 6.e0*u15*u20*u28 + 2.e0*u1*u2*u3 + 2.e0*u15*u2*u3 \
                    + 2.e0*u1*u20*u3 + 2.e0*u15*u20*u3 + 2.e0*u10*u2*u38 + 6.e0*u10 \
                    *u20*u38 + 6.e0*u2*u28*u38 + 2.4e1*u20*u28*u38 + 4.e0*u2*u3*u38 \
                    + 6.e0*u20*u3*u38 + 2.e0*u10*u2*u5 + 2.e0*u10*u20*u5 + 4.e0*u2 \
                    *u28*u5 + 6.e0*u20*u28*u5 + 6.e0*u2*u3*u5 + 4.e0*u20*u3*u5 + 2.e0 \
                    *u1*u10*u7 + 6.e0*u10*u15*u7 + 2.e0*u1*u28*u7 + 4.e0*u15*u28*u7 \
                    + u1*u3*u7 + 2.e0*u15*u3*u7 + 4.e0*u10*u38*u7 + 6.e0*u28*u38*u7 \
                    + 2.e0*u3*u38*u7 + 2.e0*u10*u5*u7 + 2.e0*u28*u5*u7 + 2.e0*u3*u5*u7 \
                    + u60*(2.e0*u15*u20 + 6.e0*u20*u38 + 2.e0*u20*u5 + u2*(u15 + 2.e0 \
                    *u5 + u63) + 2.e0*u15*u7 + 2.e0*u38*u7 + u5*u7 + 2.e0*u1*(u2 + 2.e0 \
                    *u20 + u7)) + (2.e0*u10*u15 + 2.e0*u15*u28 + u15*u3 + 2.e0*u1*(u10 \
                    + 2.e0*u28 + u3) + 2.e0*u10*u38 + 6.e0*u28*u38 + 2.e0*u3*u38 + u10 \
                    *u5 + 2.e0*u28*u5 + 2.e0*u3*u5 + 2.e0*u60*(3.e0*u1 + u15 + u5 + u63))*v[0,0])
        else:
            print('Invalid value for n!')
            return np.nan

    def vol_simplex(self, v):
        """
        Computes volume of tetrahedron.

        Args:
            v (4x3 array of ints): coordinates of the four tetrahedron vertices

        Returns:
            Volume of tetrahedron defined by four given vertices.
        """
        v14 = v[0]-v[3]
        v24 = v[1]-v[3]
        v34 = v[2]-v[3]
        return 1./6*np.abs(np.dot(v14,np.cross(v24,v34)))

    def do_binning_integral(self):
        """
        Tabulate values of interpolation kernels for all configurations.
        """
        if self.delaunay_grids is None:
            self.generate_Delaunay_triangularisation()
        self.alpha_vertices = []
        for n,tri in enumerate(self.tri):
            alpha_tri = np.zeros([self.delaunay_grids[n].simplices.shape[0],5])
            for i,simp in enumerate(self.delaunay_grids[n].simplices):
                vol = self.vol_simplex(self.delaunay_grids[n].points[simp]) # previously equivalent to vol = 1./6
                for j in range(5):
                    alpha_tri[i,j] = 6*vol*self.alpha(j,self.delaunay_grids[n].points[simp])
            self.alpha_vertices.append(alpha_tri)

    def compute_Delaunay_matrix(self):
        """
        Build Delaunay matrix that converts bispectra evaluated at the list of unique vertices to the bin averaged bispectra at the target configurations.
        The Delaunay matrix is stored as a sparse matrix, using the the compressed sparse row (CSR) format.
        """
        self.do_binning_integral()
        if self.vertex_lookup is None:
            self.generate_vertex_lookup()
        self.DM_v = []
        self.DM_col_index = []
        self.DM_row_index = [0]
        for n in range(self.Ntri):
            DM_v = np.zeros(self.unique_vertices.shape[0])
            for i in range(self.vertex_lookup[n].shape[0]): # sum over simplices
                DM_v[self.vertex_lookup[n][i]] += self.alpha_vertices[n][i,1:]
            DM_v /= np.sum(self.alpha_vertices[n][:,0])
            unique_vertices = np.unique(self.vertex_lookup[n])
            self.DM_v += DM_v[unique_vertices].tolist()
            self.DM_col_index += unique_vertices.tolist()
            self.DM_row_index.append(self.DM_row_index[n] + len(unique_vertices))

    def init_vertices(self, table=None):
        """
        Initialise table evaluated at list of unique vertices.

        Args:
            table (str or float): array with number of rows corresponding to unique vertices, any number of columns corresponding to different bispectrum contributions,
                                  or filename pointing to array
        """
        if table is not None:
            if isinstance(table, str):
                self.Btable_vertices_unique = np.loadtxt(table)
            else:
                self.Btable_vertices_unique = np.copy(table)
        else:
            sys.exit('Please provide a valid table.')

    def Btable_delaunay(self, tri_indices, table=None):
        """
        Compute bin average of bispectrum table.

        Args:
            tri_indices (int): single value or array of triangle index for which to compute the bin average
            table (str or float): array with number of rows corresponding to unique vertices, any number of columns corresponding to different bispectrum contributions,
                                  or filename pointing to array

        Returns:
            Bin averaged bispectrum table.
        """
        if self.DM_v is None:
            self.compute_Delaunay_matrix()
        if self.Btable_vertices_unique is None:
            if table is None:
                sys.exit('Vertices not initialised. Please provide a table.')
            else:
                self.init_vertices(table=table)
        if not isinstance(tri_indices,list) and not isinstance(tri_indices,np.ndarray):
            tri_indices = [tri_indices]
        bisp = np.zeros([len(tri_indices),self.Btable_vertices_unique.shape[1]])
        for n, tri_index in enumerate(tri_indices):
            for r in range(self.DM_row_index[tri_index],self.DM_row_index[tri_index+1]):
                bisp[n,:] += self.DM_v[r]*self.Btable_vertices_unique[self.DM_col_index[r]]
        return bisp

    def export_vertices(self,fname):
        """
        Export list of unique vertices to file.

        Args:
            fname (str): filename
        """
        np.savetxt(fname,self.unique_vertices)

    def export_Delaunay_matrix_as_CSR(self, fname_base):
        """
        Export Delaunay matrix in compressed sparse row (CSR) format. Creates three files ending in _v.dat, _col_index.dat and _row_index.dat,
        storing the non-zero values, the corresponding column indices and the beginning of the row.

        Args:
            fname_base (str): filename root
        """
        np.savetxt(fname_base + '_v.dat',np.array(self.DM_v))
        np.savetxt(fname_base + '_col_index.dat',np.array(self.DM_col_index))
        np.savetxt(fname_base + '_row_index.dat',np.array(self.DM_row_index))

    def init_Delaunay_matrix_from_file(self, fname_base, generate_vertices=True):
        """
        Initialise sparse matrix components from previously stored files.

        Args:
            fname_base (str): filename root
            generate_vertices (bool): if true, also generate list of vertices
        """
        self.DM_v = np.loadtxt(fname_base + '_v.dat')
        self.DM_col_index = np.loadtxt(fname_base + '_col_index.dat').astype(int)
        self.DM_row_index = np.loadtxt(fname_base + '_row_index.dat').astype(int)

        if generate_vertices:
            self.generate_Delaunay_triangularisation()

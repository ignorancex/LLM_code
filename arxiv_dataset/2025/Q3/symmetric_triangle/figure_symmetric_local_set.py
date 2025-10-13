import numpy as np
import vtk
import pyvista as pv
from numpy.polynomial import Polynomial
from triangle_inequalities import valid_distribution


# ------------------------------------------------------ variable step 1d array
def non_uniform_space(start, end, n_points, end_start_ratio):
    """
    Generates an array of `n_points` between `start` and `end` with a variable step rate.

    The step size changes linearly, and the ratio of the last step size to the first
    step size is controlled by the `end_start_ratio` parameter. This allows for
    non-uniform spacing of points, where the step size can either increase or decrease
    across the sequence.

    Parameters:
    -----------
    start : float
        The starting value of the sequence.
    end : float
        The ending value of the sequence.
    n_points : int
        The number of points in the sequence.
    end_start_ratio : float
        The ratio of the last step size to the first step size.
        - If `end_start_ratio > 1`, the step size increases across the sequence.
        - If `end_start_ratio < 1`, the step size decreases across the sequence.
        - If `end_start_ratio == 1`, the step size is constant (equivalent to `numpy.linspace`).

    Returns:
    --------
    numpy.ndarray
        An array of `n_points` values between `start` and `end` with a variable step rate.

    Notes:
    ------
    - The step size changes linearly, meaning the difference between consecutive steps
      is constant.
    - The function solves a system of linear equations to determine the coefficients
      for the step size formula.

    Examples:
    ---------
    >>> # Increasing step size
    >>> arr = non_uniform_space(0, 10, 5, 2)
    >>> print(arr)
    [ 0.          1.66666667  3.88888889  6.66666667 10.        ]

    >>> # Decreasing step size
    >>> arr = non_uniform_space(0, 10, 5, 0.5)
    >>> print(arr)
    [0.         3.33333333 6.11111111 8.33333333 10.        ]

    >>> # Constant step size (equivalent to numpy.linspace)
    >>> arr = non_uniform_space(0, 10, 5, 1)
    >>> print(arr)
    [0.  2.5 5.  7.5 10. ]
    """
    n = n_points  # number of points
    r = end_start_ratio  # shrinking ratio (end step)/(start step)
    alpha, beta = np.linalg.solve([[n*(n - 1)/2, n - 1],
                                   [n - 1 - r, 1 - r]],
                                  [end - start, 0])
    k = np.linspace(0, n-1, n)
    return start + k*beta + k*(k + 1)*alpha/2


# -------------------------------------------------------- Full behavior space
behavior_space = pv.PolyData([[1, 1, 1],         # D+
                              [-1, 1, -1],       # D-
                              [1/3, -1/3, -1],   # W
                              [-1/3, -1/3, 1]],  # W flipped
                             [3, 0, 1, 2, 3, 0, 1, 3, 3, 0, 2, 3, 3, 1, 2, 3])

# ---------------------------------------------------------------- GHZ surface
anum = np.linspace(0,1,101)
bnum = np.linspace(0,1,101)

anum, bnum = np.meshgrid(anum, bnum, indexing='ij')
anum[anum + bnum > 1] = np.nan
bnum[anum + bnum > 1] = np.nan

e1 = 4*anum*bnum + 2*bnum**2 - 1
e2 = 4*anum**2*bnum + 12*anum*bnum**2 - 8*anum*bnum + 4*bnum**3 - 4*bnum**2 + 1
e3 = -12*anum**2*bnum - 12*anum*bnum**2 + 12*anum*bnum - 4*bnum**3 + 6*bnum**2 - 1

e1 = e1[~np.isnan(anum)]
e2 = e2[~np.isnan(anum)]
e3 = e3[~np.isnan(anum)]

GHZ_boundary_behaviors = pv.PolyData(np.vstack((e1, e2, e3)).T)
GHZ_boundary_behaviors_surface = GHZ_boundary_behaviors.delaunay_2d()
GHZ_boundary_behaviors_surface = GHZ_boundary_behaviors.reconstruct_surface()
GHZ_boundary_behaviors_surface = GHZ_boundary_behaviors_surface.clip((1, 1, -1),
                                                                     (1, 0, 0))
GHZ_boundary_behaviors_surface = GHZ_boundary_behaviors_surface.clip((-1, 1, 1),
                                                                     (0, 0, 1))

# --------------------------------------------------------- Flipped GHZ surface
GHZ_boundary_behaviors2 = pv.PolyData(np.vstack((-e1, e2, -e3)).T)
GHZ_boundary_behaviors_surface2 = GHZ_boundary_behaviors2.delaunay_2d()
GHZ_boundary_behaviors_surface2 = GHZ_boundary_behaviors2.reconstruct_surface()
GHZ_boundary_behaviors_surface2 = GHZ_boundary_behaviors_surface2.clip((1,1,-1), 
                                                                       (1,0,0))
GHZ_boundary_behaviors_surface2 = GHZ_boundary_behaviors_surface2.clip((-1,1,1), 
                                                                       (0,0,1))

# ------------------------------------------------------- Clipping GHZ surfaces
GHZ1 = GHZ_boundary_behaviors_surface.clip_surface(
    GHZ_boundary_behaviors_surface2, invert=False)
GHZ2 = GHZ_boundary_behaviors_surface2.clip_surface(GHZ_boundary_behaviors_surface)

# ---------------------------------------------------------------- 322 W models
N = 20
a_max = Polynomial([1, -9, 24, -24, 9]).roots()[0].real
a = np.broadcast_to(np.linspace(0, a_max, N), (N, N)).T
b = np.empty_like(a)
for i, a_ in enumerate(a[:,0]):
    b_min = (1 + a_ + np.sqrt((1 + 5*a_**2 - 2*a_**3)/(1 - 2*a_)))/4
    b_max = (-1 + 2*a_ + np.sqrt(5 - 8*a_ + 4*a_**2))/2
    b[i] = np.linspace(b_min, b_max, N)

e1 = (1 - 2*a)*(2*b - 1)
e2 = 1 - 4*a - 4*b + 8*a*b + 4*b**2 - 8*a*b**2
e3 = -(1 - 2*a)*(1 - 4*a + 2*b + 8*a*b - 4*b**2)

e1, e2, e3 = e1.flatten(), e2.flatten(), e3.flatten()

# e1 = np.linspace(0, np.sqrt(5)-2, 60)
# e2 = np.linspace(-1/3, 9-4*np.sqrt(5), 100)
# e1, e2 = np.meshgrid(e1, e2, indexing='ij')
# valid = 1 + e2 >= 2*np.abs(e1)
# e1, e2 = e1[valid], e2[valid]
# e3 = -2*(2*e1 - e2 - 1)*(4*e1**3 - 4*e1**2 - e1*e2 - e1 + e2**2 + 2*e2 + (e1 - e2 - 1)*np.sqrt(-4*e1**2 + e2**2 + 2*e2 + 1) + 1)/(2*e1 - e2 + np.sqrt(-4*e1**2 + e2**2 + 2*e2 + 1) - 1)**2
# e3[e1==0] = -1
# e2[e1==0] = 0
# valid = valid_distribution(e1, e2, e3)
# e1, e2, e3 = e1[valid], e2[valid], e3[valid]
points_322 = np.vstack((e1, e2, e3)).T
W_322 = pv.PolyData(points_322)
W_322 = W_322.delaunay_2d(tol=1e-5, alpha=0.1, bound=True, progress_bar=True)

# ---------------------------------------------------------------- 422 W models
N = 21
e1_array = non_uniform_space(1/3, 1/2, N, 30)
e1_array = np.broadcast_to(e1_array, (N, N)).T
q = np.empty_like(e1_array)
for i, e1 in enumerate(e1_array[:,0]):
    d = np.sqrt((1-e1)/2)
    e2 = Polynomial(((2*e1 - 1)**2*(4096*e1**25 - 40960*e1**24 + 257024*e1**23 - 1168384*e1**22 + 3921152*e1**21 - 9820672*e1**20 + 19352128*e1**19 - 35225536*e1**18 + 74516656*e1**17 - 178218752*e1**16 + 386085556*e1**15 - 671145868*e1**14 + 904538840*e1**13 - 933499608*e1**12 + 724848661*e1**11 - 407931861*e1**10 + 152953347*e1**9 - 29599147*e1**8 - 1446650*e1**7 + 1777890*e1**6 - 106338*e1**5 - 51182*e1**4 + 641*e1**3 + 943*e1**2 + 71*e1 + 1), -2*(2*e1 - 1)*(20480*e1**24 - 26624*e1**23 - 527360*e1**22 + 3209728*e1**21 - 9007360*e1**20 + 18382208*e1**19 - 57703104*e1**18 + 255773536*e1**17 - 900509968*e1**16 + 2286521176*e1**15 - 4299714908*e1**14 + 6137140762*e1**13 - 6714586154*e1**12 + 5601727181*e1**11 - 3481643369*e1**10 + 1527261521*e1**9 - 411254489*e1**8 + 31409034*e1**7 + 19797434*e1**6 - 7001220*e1**5 + 687108*e1**4 + 76193*e1**3 - 22533*e1**2 + 745*e1 + 223), 90112*e1**24 + 36864*e1**23 - 2674688*e1**22 + 10257408*e1**21 - 2399232*e1**20 - 67424256*e1**19 + 37204608*e1**18 + 1152374912*e1**17 - 6026605824*e1**16 + 17350357168*e1**15 - 34451561936*e1**14 + 50460275056*e1**13 - 55641835968*e1**12 + 45895091069*e1**11 - 27301467461*e1**10 + 10556212959*e1**9 - 1658238007*e1**8 - 686935502*e1**7 + 478329438*e1**6 - 103622770*e1**5 - 1074062*e1**4 + 4269809*e1**3 - 555801*e1**2 - 17837*e1 + 6101, -729088*e1**22 + 4634624*e1**21 - 11909120*e1**20 - 7633920*e1**19 + 107279872*e1**18 - 126346752*e1**17 - 867555968*e1**16 + 4710416480*e1**15 - 12377991392*e1**14 + 20742551280*e1**13 - 22268054240*e1**12 + 11376413384*e1**11 + 7725071448*e1**10 - 22773516824*e1**9 + 25151063688*e1**8 - 17430231120*e1**7 + 8119680400*e1**6 - 2497965120*e1**5 + 464366128*e1**4 - 38724696*e1**3 - 1527688*e1**2 + 482024*e1 - 19160, 239104*e1**21 - 1102336*e1**20 + 1242112*e1**19 + 16131712*e1**18 - 43681664*e1**17 - 126848512*e1**16 + 1168480472*e1**15 - 3840545352*e1**14 + 7044753704*e1**13 - 5756144448*e1**12 - 6652288886*e1**11 + 30264043526*e1**10 - 53346579954*e1**9 + 60610400826*e1**8 - 48436723844*e1**7 + 27701832564*e1**6 - 11191339196*e1**5 + 3071159604*e1**4 - 530371262*e1**3 + 49258798*e1**2 - 1432346*e1 - 54542, -276992*e1**19 + 2075904*e1**18 - 14173440*e1**17 + 20513664*e1**16 + 103670752*e1**15 - 770593760*e1**14 + 2463469256*e1**13 - 4359851312*e1**12 + 2726655316*e1**11 + 7436492636*e1**10 - 25857891836*e1**9 + 42706972388*e1**8 - 46086717208*e1**7 + 34696298936*e1**6 - 18411541040*e1**5 + 6763190808*e1**4 - 1643210460*e1**3 + 242371468*e1**2 - 18449116*e1 + 477940, 42368*e1**18 - 258176*e1**17 + 3621632*e1**16 - 6251728*e1**15 + 1289008*e1**14 + 136980016*e1**13 - 714921024*e1**12 + 1772299586*e1**11 - 2085218770*e1**10 - 941497034*e1**9 + 8238330954*e1**8 - 15755567356*e1**7 + 17613546044*e1**6 - 12982248484*e1**5 + 6454923540*e1**4 - 2122224982*e1**3 + 433777350*e1**2 - 48347362*e1 + 2154498, -59776*e1**16 - 70240*e1**15 - 1917472*e1**14 - 9380048*e1**13 + 28379424*e1**12 + 16867080*e1**11 - 279458408*e1**10 + 734910248*e1**9 - 629758968*e1**8 - 845892752*e1**7 + 2885394128*e1**6 - 3642971008*e1**5 + 2679690032*e1**4 - 1232752024*e1**3 + 347338040*e1**2 - 53998936*e1 + 3434920, 4356*e1**15 + 21812*e1**14 + 987308*e1**13 + 1396672*e1**12 - 6210143*e1**11 + 21651047*e1**10 - 22295541*e1**9 - 87149631*e1**8 + 221289198*e1**7 - 107942342*e1**6 - 213655678*e1**5 + 386501762*e1**4 - 289526851*e1**3 + 118724443*e1**2 - 26090889*e1 + 2386637, -2*(e1 + 1)*(9*e1 - 7)*(594*e1**11 + 4896*e1**10 + 22391*e1**9 + 71269*e1**8 - 17436*e1**7 - 483192*e1**6 + 367766*e1**5 + 743690*e1**4 - 1378574*e1**3 + 934624*e1**2 - 304181*e1 + 40713), (e1 + 1)**3*(9*e1 - 7)**4*(e1**2 + 4*e1 - 1)*(e1**2 + 12*e1 - 9))).roots()
    e2[np.iscomplex(e2)] = np.nan
    e2 = np.sort(e2)[2].real
    q_min = (2*d**2 - 2*d + 1)*(4*d**4 - 4*d**3 - 2*d*e2 + 2*d + e2 - 1)/(d*(8*d**4 - 12*d**3 - d**2*e2 + 5*d**2 - d*e2 + d + e2 - 1))
    q_max = 1
    q[i] = np.linspace(q_min, q_max, N)
q, e1_array = q.flatten(), e1_array.flatten()

d = np.sqrt((1-e1_array)/2)
e2_array = (-8*d**6 + 8*d**5*q + 16*d**5 - 12*d**4*q - 12*d**4 + 5*d**3*q + d**2*q + 6*d**2 - d*q - 4*d + 1)/(d**3*q - 4*d**3 + d**2*q + 6*d**2 - d*q - 4*d + 1)
e3_array = (-8*d**6*q + 30*d**5*q - 38*d**4*q + 15*d**3*q + d**2*q + 4*d**2 - d*q - 4*d + 1)/(d**3*q - 4*d**3 + d**2*q + 6*d**2 - d*q - 4*d + 1)

valid = ~(~valid_distribution(e1_array, e2_array, e3_array) & np.isclose(e1, 1/3))
e1, e2, e3 = e1_array[valid], e2_array[valid], e3_array[valid]

points_422 = np.vstack((e1, e2, e3)).T
W_422 = pv.PolyData(points_422)
W_422 = W_422.delaunay_2d(tol=1e-5, alpha=0.05, bound=True, progress_bar=True)

# ------------------------------------------------- 333 W models no constraints
N = 50
e = np.broadcast_to(np.linspace(1/3, 1, N), (N, N)).T
a = np.empty_like(e)
for i, e_ in enumerate(e[:,0]):
    a_min = np.maximum(1 + 12*e_ + 3*e_**2 - np.sqrt((1 + 12*e_ + 3*e_**2)**2 - 192*e_**2), 
                       12*e_*(2 + e_ - np.sqrt(2 + 2*e_ + e_**2)))/(24*e_)
    a_max = Polynomial([4*e_**2*(1 + e_)**2, 
                        -(5 + 16*e_ + 50*e_**2 + 24*e_**3 + e_**4)*e_, 
                        1 + 12*e_ + 34*e_**2 + 84*e_**3 + 45*e_**4, 
                        -12*e_*(1 + 2*e_ + 5*e_**2), 
                        16*e_**2]).roots().min().real
    a[i] = np.linspace(a_min, a_max, N)
e, a = e.flatten(), a.flatten()

e1, e2, e3 = (-e*(3*a - 1)/(a - e), 
              (4*a**3*e - 4*a**2*e**2 - 6*a*e - a + 3*e)/(a - e), 
              3*e*(4*a**3 - 4*a**2*e - 8*a**2 + 2*a*e + 5*a - 1)/(a - e))

points_333_0 = np.vstack((e1, e2, e3)).T
points_333_0 = points_333_0[~np.any(np.isinf(points_333_0), axis=1)]

# intersection with cyan surface
e = np.linspace(1/3, 1, N)
a = np.empty_like(e)
for i, e_ in enumerate(e):
    a[i] = Polynomial([4*e_**2*(1 + e_)**2, 
                        -5*e_ - 16*e_**2 - 50*e_**3 - 24*e_**4 - e_**5, 
                        1 + 12*e_ + 34*e_**2 + 84*e_**3 + 45*e_**4, 
                        -12*e_ - 24*e_**2 - 60*e_**3, 
                        16*e_**2]).roots()[0].real
e1_border = (1-3*a)*e/(a-e)
e1_border[0] = 1/3
e2_border = (-a + 3*e - 6*a*e + 4*a**3*e - 4*a**2*e**2)/(a - e)
e2_border[0] = -5/27
e3_border = 3*e*(4*a**3 - 4*a**2*e - 8*a**2 + 2*a*e + 5*a - 1)/(a - e)
e3_border[0] = -5/9
border = np.vstack((e1_border, e2_border, e3_border)).T
points_333_0 = np.vstack((points_333_0, border))

# intersection with D0 W Wflip plane
e_max = Polynomial([1, -12, 6, 12, 9]).roots()[3].real
e = np.linspace(1/3, e_max, N)
a = (1/24)*(3*e**2 + 12*e - np.sqrt(9*e**4 + 72*e**3 - 42*e**2 + 24*e + 1) + 1)/e
e1_border = (1-3*a)*e/(a-e)
e2_border = (-a + 3*e - 6*a*e + 4*a**3*e - 4*a**2*e**2)/(a - e)
e3_border = 3*e*(4*a**3 - 4*a**2*e - 8*a**2 + 2*a*e + 5*a - 1)/(a - e)
border = np.vstack((e1_border, e2_border, e3_border)).T
points_333_0 = np.vstack((points_333_0, border[1:]))

W_333_0 = pv.PolyData(np.append(points_333_0, [[1/3, -5/27, -5/9]], axis=0))
W_333_0 = W_333_0.delaunay_2d(tol=1e-5, alpha=0.06, bound=True, progress_bar=True)

# ---------------------------------------------------------------- 332 W models
N = 36
N1 = round(5/12*N)
N2 = round(7/12*N)
e1_array = np.hstack((np.linspace(np.sqrt(5)-2, 1/3, N1), 
                      non_uniform_space(1/3, 1/2, N2, 30)[1:]))

e1_array = np.broadcast_to(e1_array, (N-1, N-1)).T
e2_array = np.empty_like(e1_array)

for i, e1 in enumerate(e1_array[:,0]):
    if e1 > 1/3:
        roots = Polynomial(((2*e1 - 1)**2*(4096*e1**25 - 40960*e1**24 + 257024*e1**23 - 1168384*e1**22 + 3921152*e1**21 - 9820672*e1**20 + 19352128*e1**19 - 35225536*e1**18 + 74516656*e1**17 - 178218752*e1**16 + 386085556*e1**15 - 671145868*e1**14 + 904538840*e1**13 - 933499608*e1**12 + 724848661*e1**11 - 407931861*e1**10 + 152953347*e1**9 - 29599147*e1**8 - 1446650*e1**7 + 1777890*e1**6 - 106338*e1**5 - 51182*e1**4 + 641*e1**3 + 943*e1**2 + 71*e1 + 1), -2*(2*e1 - 1)*(20480*e1**24 - 26624*e1**23 - 527360*e1**22 + 3209728*e1**21 - 9007360*e1**20 + 18382208*e1**19 - 57703104*e1**18 + 255773536*e1**17 - 900509968*e1**16 + 2286521176*e1**15 - 4299714908*e1**14 + 6137140762*e1**13 - 6714586154*e1**12 + 5601727181*e1**11 - 3481643369*e1**10 + 1527261521*e1**9 - 411254489*e1**8 + 31409034*e1**7 + 19797434*e1**6 - 7001220*e1**5 + 687108*e1**4 + 76193*e1**3 - 22533*e1**2 + 745*e1 + 223), 90112*e1**24 + 36864*e1**23 - 2674688*e1**22 + 10257408*e1**21 - 2399232*e1**20 - 67424256*e1**19 + 37204608*e1**18 + 1152374912*e1**17 - 6026605824*e1**16 + 17350357168*e1**15 - 34451561936*e1**14 + 50460275056*e1**13 - 55641835968*e1**12 + 45895091069*e1**11 - 27301467461*e1**10 + 10556212959*e1**9 - 1658238007*e1**8 - 686935502*e1**7 + 478329438*e1**6 - 103622770*e1**5 - 1074062*e1**4 + 4269809*e1**3 - 555801*e1**2 - 17837*e1 + 6101, -729088*e1**22 + 4634624*e1**21 - 11909120*e1**20 - 7633920*e1**19 + 107279872*e1**18 - 126346752*e1**17 - 867555968*e1**16 + 4710416480*e1**15 - 12377991392*e1**14 + 20742551280*e1**13 - 22268054240*e1**12 + 11376413384*e1**11 + 7725071448*e1**10 - 22773516824*e1**9 + 25151063688*e1**8 - 17430231120*e1**7 + 8119680400*e1**6 - 2497965120*e1**5 + 464366128*e1**4 - 38724696*e1**3 - 1527688*e1**2 + 482024*e1 - 19160, 239104*e1**21 - 1102336*e1**20 + 1242112*e1**19 + 16131712*e1**18 - 43681664*e1**17 - 126848512*e1**16 + 1168480472*e1**15 - 3840545352*e1**14 + 7044753704*e1**13 - 5756144448*e1**12 - 6652288886*e1**11 + 30264043526*e1**10 - 53346579954*e1**9 + 60610400826*e1**8 - 48436723844*e1**7 + 27701832564*e1**6 - 11191339196*e1**5 + 3071159604*e1**4 - 530371262*e1**3 + 49258798*e1**2 - 1432346*e1 - 54542, -276992*e1**19 + 2075904*e1**18 - 14173440*e1**17 + 20513664*e1**16 + 103670752*e1**15 - 770593760*e1**14 + 2463469256*e1**13 - 4359851312*e1**12 + 2726655316*e1**11 + 7436492636*e1**10 - 25857891836*e1**9 + 42706972388*e1**8 - 46086717208*e1**7 + 34696298936*e1**6 - 18411541040*e1**5 + 6763190808*e1**4 - 1643210460*e1**3 + 242371468*e1**2 - 18449116*e1 + 477940, 42368*e1**18 - 258176*e1**17 + 3621632*e1**16 - 6251728*e1**15 + 1289008*e1**14 + 136980016*e1**13 - 714921024*e1**12 + 1772299586*e1**11 - 2085218770*e1**10 - 941497034*e1**9 + 8238330954*e1**8 - 15755567356*e1**7 + 17613546044*e1**6 - 12982248484*e1**5 + 6454923540*e1**4 - 2122224982*e1**3 + 433777350*e1**2 - 48347362*e1 + 2154498, -59776*e1**16 - 70240*e1**15 - 1917472*e1**14 - 9380048*e1**13 + 28379424*e1**12 + 16867080*e1**11 - 279458408*e1**10 + 734910248*e1**9 - 629758968*e1**8 - 845892752*e1**7 + 2885394128*e1**6 - 3642971008*e1**5 + 2679690032*e1**4 - 1232752024*e1**3 + 347338040*e1**2 - 53998936*e1 + 3434920, 4356*e1**15 + 21812*e1**14 + 987308*e1**13 + 1396672*e1**12 - 6210143*e1**11 + 21651047*e1**10 - 22295541*e1**9 - 87149631*e1**8 + 221289198*e1**7 - 107942342*e1**6 - 213655678*e1**5 + 386501762*e1**4 - 289526851*e1**3 + 118724443*e1**2 - 26090889*e1 + 2386637, -2*(e1 + 1)*(9*e1 - 7)*(594*e1**11 + 4896*e1**10 + 22391*e1**9 + 71269*e1**8 - 17436*e1**7 - 483192*e1**6 + 367766*e1**5 + 743690*e1**4 - 1378574*e1**3 + 934624*e1**2 - 304181*e1 + 40713), (e1 + 1)**3*(9*e1 - 7)**4*(e1**2 + 4*e1 - 1)*(e1**2 + 12*e1 - 9))).roots()
        roots[np.iscomplex(roots)] = np.nan
        e2_max = np.sort(roots.real)[2]
    else:
        e2_max = e1**2
    roots = Polynomial((16*e1**11 + 112*e1**10 + 208*e1**9 - 80*e1**8 - 464*e1**7 - 176*e1**6 + 240*e1**5 + 144*e1**4, 5*e1**11 - 221*e1**10 - 901*e1**9 - 43*e1**8 + 3522*e1**7 + 3054*e1**6 - 1562*e1**5 - 1766*e1**4 - 23*e1**3 - e1**2 - 17*e1 + 1, 26*e1**10 + 1040*e1**9 + 694*e1**8 - 9632*e1**7 - 15356*e1**6 + 1472*e1**5 + 8556*e1**4 + 800*e1**3 - 62*e1**2 + 176*e1 - 2, 26*e1**10 - 240*e1**9 - 842*e1**8 + 11968*e1**7 + 34660*e1**6 + 10720*e1**5 - 18548*e1**4 - 6464*e1**3 + 98*e1**2 - 624*e1 - 34, -36*e1**9 + 676*e1**8 - 4944*e1**7 - 37648*e1**6 - 35256*e1**5 + 9688*e1**4 + 23280*e1**3 + 2544*e1**2 + 572*e1 + 164, -44*e1**9 - 756*e1**8 - 4224*e1**7 + 13184*e1**6 + 43400*e1**5 + 38200*e1**4 - 46304*e1**3 - 14624*e1**2 + 2052*e1 - 164, 432*e1**8 + 5408*e1**7 + 10976*e1**6 - 18592*e1**5 - 96256*e1**4 + 56544*e1**3 + 37280*e1**2 - 7520*e1 - 560, -48*e1**8 - 1728*e1**7 - 11968*e1**6 - 8896*e1**5 + 111584*e1**4 - 45632*e1**3 - 55488*e1**2 + 12224*e1 + 2000, 3584*e1**6 + 12864*e1**5 - 79808*e1**4 + 26496*e1**3 + 52608*e1**2 - 12736*e1 - 3008, 64*e1**7 - 256*e1**6 - 5568*e1**5 + 38528*e1**4 - 12096*e1**3 - 32768*e1**2 + 9408*e1 + 2688, 32*e1**6 + 1344*e1**5 - 12832*e1**4 + 4480*e1**3 + 13280*e1**2 - 4800*e1 - 1504, -32*e1**6 - 320*e1**5 + 2848*e1**4 - 1152*e1**3 - 3296*e1**2 + 1472*e1 + 480, 64*e1**5 - 320*e1**4 + 128*e1**3 + 384*e1**2 - 192*e1 - 64)).roots()
    roots[np.iscomplex(roots)] = np.nan
    roots = np.sort(roots.real)
    c = roots[1]
    e2_min = (-np.sqrt(c)*(2*c**2 - 2*c*e1 - 2*c + e1 + 1)*np.sqrt(4*c**5*e1**2 - 8*c**5*e1 + 4*c**5 - 16*c**4*e1**2 + 32*c**4*e1 - 16*c**4 - 4*c**3*e1**3 + 12*c**3*e1**2 - 44*c**3*e1 + 36*c**3 + 8*c**2*e1**3 + 8*c**2*e1**2 + 24*c**2*e1 - 40*c**2 + 5*c*e1**4 - 4*c*e1**3 - 18*c*e1**2 - 4*c*e1 + 21*c - 4*e1**4 + 8*e1**2 - 4) - 4*c**5*e1 + 4*c**5 + 16*c**4*e1 - 12*c**4 - 8*c**3*e1**2 - 30*c**3*e1 + 22*c**3 + 4*c**2*e1**3 + 16*c**2*e1**2 + 26*c**2*e1 - 22*c**2 - 3*c*e1**3 - 13*c*e1**2 - 11*c*e1 + 11*c + 4*e1**2 + 2*e1 - 2)/(4*c**4 - 2*c**3*e1 - 14*c**3 + 6*c**2*e1 + 18*c**2 - 6*c*e1 - 10*c + 2*e1 + 2)
    e2_array[i] = np.linspace(e2_min, e2_max, N-1)

e1_array, e2_array = e1_array.flatten(), e2_array.flatten()
e3_array = np.empty_like(e1_array)

for i, (e1, e2) in enumerate(zip(e1_array, e2_array)):
    roots = Polynomial((256*e1**8 - 256*e1**7*e2 - 882*e1**7 + 64*e1**6*e2**2 + 504*e1**6*e2 + 1013*e1**6 + 212*e1**5*e2**2 + 161*e1**5*e2 - 252*e1**5 - 184*e1**4*e2**3 - 766*e1**4*e2**2 - 919*e1**4*e2 - 345*e1**4 + 30*e1**3*e2**4 + 258*e1**3*e2**3 + 432*e1**3*e2**2 + 530*e1**3*e2 + 270*e1**3 - 7*e1**2*e2**4 + 118*e1**2*e2**3 + 364*e1**2*e2**2 + 98*e1**2*e2 - 61*e1**2 - 3*e1*e2**5 - 52*e1*e2**4 - 290*e1*e2**3 - 404*e1*e2**2 - 147*e1*e2 + e2**5 + 29*e2**4 + 98*e2**3 + 98*e2**2 + 29*e2 + 1, 88*e1**6 - 232*e1**5*e2 - 299*e1**5 + 168*e1**4*e2**2 + 476*e1**4*e2 + 409*e1**4 - 40*e1**3*e2**3 - 138*e1**3*e2**2 - 192*e1**3*e2 - 278*e1**3 - 36*e1**2*e2**3 - 230*e1**2*e2**2 - 152*e1**2*e2 + 82*e1**2 + 13*e1*e2**4 + 112*e1*e2**3 + 250*e1*e2**2 + 136*e1*e2 + e1 - 3*e2**4 - 36*e2**3 - 50*e2**2 - 36*e2 - 3, 36*e1**5 - 24*e1**4*e2 - 62*e1**4 + 4*e1**3*e2**2 - 66*e1**3*e2 + 16*e1**3 + 86*e1**2*e2**2 + 154*e1**2*e2 + 12*e1**2 - 22*e1*e2**3 - 88*e1*e2**2 - 62*e1*e2 - 4*e1 + 2*e2**3 - 2*e2**2 - 2*e2 + 2, -8*e1**4 + 8*e1**3*e2 + 42*e1**3 - 52*e1**2*e2 - 42*e1**2 + 18*e1*e2**2 + 32*e1*e2 + 6*e1 + 2*e2**2 + 12*e2 + 2, -2*e1**3 + 9*e1**2 - 7*e1*e2 - 4*e1 - 3*e2 - 3, e1 + 1)).roots()
    roots[np.iscomplex(roots)] = np.nan
    e3_array[i] = np.sort(roots.real)[0]

points = np.vstack((e1_array, e2_array, e3_array)).T

W_332 = pv.PolyData(points)
W_332 = W_332.delaunay_2d(alpha=0.04, progress_bar=True)

# --------------------------------------------------- 333 W models 1 constraint

# The following block of code takes too long to run, so instead I commented it and
# just read the file with the point coordinates calculated with this procedure.

# N = 15
# e1 = np.broadcast_to(np.linspace(np.sqrt(5)-2, 1/2, N), (N, N)).T
# e2 = np.empty_like(e1)
# for i, e1_ in enumerate(e1[:, 0]):
#     if e1_ <= 1/3:
#         e2_min = np.sort(Polynomial([2048*e1_**12 - 3832*e1_**11 + 5508*e1_**10 + 12198*e1_**9 - 20061*e1_**8 + 8084*e1_**7 - 1416*e1_**6 + 9212*e1_**5 - 17250*e1_**4 + 9340*e1_**3 - 500*e1_**2 - 506*e1_ + 71,
#                                      -1024*e1_**11 + 2196*e1_**10 + 2796*e1_**9 - 23427*e1_**8 + 29532*e1_**7 - 27680*e1_**6 + 2324*e1_**5 - 25430*e1_**4 + 44052*e1_**3 - 12700*e1_**2 - 624*e1_ + 369,
#                                      1062*e1_**9 - 543*e1_**8 + 12636*e1_**7 - 2112*e1_**6 + 43568*e1_**5 + 19706*e1_**4 + 33876*e1_**3 - 31704*e1_**2 + 2682*e1_ + 477,
#                                      -729*e1_**8 - 1404*e1_**7 + 1368*e1_**6 - 156*e1_**5 - 18162*e1_**4 - 17620*e1_**3 - 20128*e1_**2 + 5100*e1_ + 243,
#                                      -2484*e1_**5 - 3456*e1_**4 - 2664*e1_**3 + 848*e1_**2 + 2268*e1_,
#                                      432*e1_**2]).roots())[2]
#     else:
#         e2_min = (-5 + 2*(3*e1_-1)*(6 - np.sqrt(2*(3*e1_-1))))/27
#     e2_max = 9-4*np.sqrt(5)
#     e2[i] = np.linspace(e2_min, e2_max, N)
# e1, e2 = e1.flatten(), e2.flatten()
#
# e3 = W5_point(e1, e2)
#
# valid = ~np.isinf(e3)
# e1, e2, e3 = e1[valid], e2[valid], e3[valid]
#
# # intersection with D0 W Wflip plane
# e = np.linspace(1/2, 1/3, 30)
# e1 = np.hstack((e1, e))
# e2 = np.hstack((e2, (36*e - 17 - np.sqrt(6*e - 2)*(6*e - 2))/27))
# e3 = np.hstack((e3, (9*e - 8 - np.sqrt(6*e - 2)*(6*e - 2))/9))
#
# # intersection with red surface
# e = np.linspace(1/3, 1, 30)
# a = np.empty_like(e)
# for i, e_ in enumerate(e):
#     a[i] = Polynomial([4*e_**2*(1 + e_)**2,
#                         -5*e_ - 16*e_**2 - 50*e_**3 - 24*e_**4 - e_**5,
#                         1 + 12*e_ + 34*e_**2 + 84*e_**3 + 45*e_**4,
#                         -12*e_ - 24*e_**2 - 60*e_**3,
#                         16*e_**2]).roots()[0].real
# e1_border = (1-3*a)*e/(a-e)
# e1_border[0] = 1/3
# e2_border = (-a + 3*e - 6*a*e + 4*a**3*e - 4*a**2*e**2)/(a - e)
# e2_border[0] = -5/27
# e3_border = 3*e*(4*a**3 - 4*a**2*e - 8*a**2 + 2*a*e + 5*a - 1)/(a - e)
# e3_border[0] = -5/9
# e1 = np.hstack((e1, e1_border))
# e2 = np.hstack((e2, e2_border))
# e3 = np.hstack((e3, e3_border))
#
# # intersection with yellow surface
# e1_border = np.linspace(np.sqrt(5)-2, 1/2, 30)
# c = np.empty_like(e1_border)
# for i, e1_ in enumerate(e1_border):
#     roots = Polynomial((16*e1_**11 + 112*e1_**10 + 208*e1_**9 - 80*e1_**8 - 464*e1_**7 - 176*e1_**6 + 240*e1_**5 + 144*e1_**4, 5*e1_**11 - 221*e1_**10 - 901*e1_**9 - 43*e1_**8 + 3522*e1_**7 + 3054*e1_**6 - 1562*e1_**5 - 1766*e1_**4 - 23*e1_**3 - e1_**2 - 17*e1_ + 1, 26*e1_**10 + 1040*e1_**9 + 694*e1_**8 - 9632*e1_**7 - 15356*e1_**6 + 1472*e1_**5 + 8556*e1_**4 + 800*e1_**3 - 62*e1_**2 + 176*e1_ - 2, 26*e1_**10 - 240*e1_**9 - 842*e1_**8 + 11968*e1_**7 + 34660*e1_**6 + 10720*e1_**5 - 18548*e1_**4 - 6464*e1_**3 + 98*e1_**2 - 624*e1_ - 34, -36*e1_**9 + 676*e1_**8 - 4944*e1_**7 - 37648*e1_**6 - 35256*e1_**5 + 9688*e1_**4 + 23280*e1_**3 + 2544*e1_**2 + 572*e1_ + 164, -44*e1_**9 - 756*e1_**8 - 4224*e1_**7 + 13184*e1_**6 + 43400*e1_**5 + 38200*e1_**4 - 46304*e1_**3 - 14624*e1_**2 + 2052*e1_ - 164, 432*e1_**8 + 5408*e1_**7 + 10976*e1_**6 - 18592*e1_**5 - 96256*e1_**4 + 56544*e1_**3 + 37280*e1_**2 - 7520*e1_ - 560, -48*e1_**8 - 1728*e1_**7 - 11968*e1_**6 - 8896*e1_**5 + 111584*e1_**4 - 45632*e1_**3 - 55488*e1_**2 + 12224*e1_ + 2000, 3584*e1_**6 + 12864*e1_**5 - 79808*e1_**4 + 26496*e1_**3 + 52608*e1_**2 - 12736*e1_ - 3008, 64*e1_**7 - 256*e1_**6 - 5568*e1_**5 + 38528*e1_**4 - 12096*e1_**3 - 32768*e1_**2 + 9408*e1_ + 2688, 32*e1_**6 + 1344*e1_**5 - 12832*e1_**4 + 4480*e1_**3 + 13280*e1_**2 - 4800*e1_ - 1504, -32*e1_**6 - 320*e1_**5 + 2848*e1_**4 - 1152*e1_**3 - 3296*e1_**2 + 1472*e1_ + 480, 64*e1_**5 - 320*e1_**4 + 128*e1_**3 + 384*e1_**2 - 192*e1_ - 64)).roots()
#     roots = roots[~np.iscomplex(roots)]
#     roots = np.sort(roots.real)
#     c[i] = roots[1]
# e2_border = (-np.sqrt(c)*(2*c**2 - 2*c*e1_border - 2*c + e1_border + 1)*np.sqrt(4*c**5*e1_border**2 - 8*c**5*e1_border + 4*c**5 - 16*c**4*e1_border**2 + 32*c**4*e1_border - 16*c**4 - 4*c**3*e1_border**3 + 12*c**3*e1_border**2 - 44*c**3*e1_border + 36*c**3 + 8*c**2*e1_border**3 + 8*c**2*e1_border**2 + 24*c**2*e1_border - 40*c**2 + 5*c*e1_border**4 - 4*c*e1_border**3 - 18*c*e1_border**2 - 4*c*e1_border + 21*c - 4*e1_border**4 + 8*e1_border**2 - 4) - 4*c**5*e1_border + 4*c**5 + 16*c**4*e1_border - 12*c**4 - 8*c**3*e1_border**2 - 30*c**3*e1_border + 22*c**3 + 4*c**2*e1_border**3 + 16*c**2*e1_border**2 + 26*c**2*e1_border - 22*c**2 - 3*c*e1_border**3 - 13*c*e1_border**2 - 11*c*e1_border + 11*c + 4*e1_border**2 + 2*e1_border - 2)/(4*c**4 - 2*c**3*e1_border - 14*c**3 + 6*c**2*e1_border + 18*c**2 - 6*c*e1_border - 10*c + 2*e1_border + 2)
# a = (2*e1_border - e2_border - 1)*(2*c**2 - 2*c*e1_border - 2*c + e1_border + 1)/(8*c**2*e1_border - 4*c*e1_border**2 - 8*c*e1_border + 2*c*e2_border + 2*c + 4*e1_border - 2*e2_border - 2)
# e3_border = (1/2)*(16*a**2*c**2*e1_border - 4*a**2*c*e1_border**2 - 22*a**2*c*e1_border + 6*a**2*c*e2_border + 6*a**2*c + 8*a**2*e1_border - 4*a**2*e2_border - 4*a**2 - 4*a*c**2*e1_border**2 - 22*a*c**2*e1_border + 6*a*c**2*e2_border + 6*a*c**2 + 18*a*c*e1_border**2 - 6*a*c*e1_border*e2_border + 20*a*c*e1_border - 10*a*c*e2_border - 10*a*c - 8*a*e1_border**2 + 4*a*e1_border*e2_border - 4*a*e1_border + 4*a*e2_border + 4*a + 8*c**2*e1_border - 4*c**2*e2_border - 4*c**2 - 8*c*e1_border**2 + 4*c*e1_border*e2_border - 4*c*e1_border + 4*c*e2_border + 4*c + 2*e1_border**3 - e1_border**2*e2_border + 3*e1_border**2 - 2*e1_border*e2_border - e2_border - 1)/(a*c*(a + c - e1_border - 1))
# e1 = np.hstack((e1, e1_border))
# e2 = np.hstack((e2, e2_border))
# e3 = np.hstack((e3, e3_border))

with np.load('W_333_1_points.npz') as data:
    e1 = data['e1']
    e2 = data['e2']
    e3 = data['e3']

W_333_1 = pv.PolyData(np.vstack((e1, e2, e3)).T)
W_333_1 = W_333_1.delaunay_2d(tol=1e-5, alpha=0.1, bound=True, progress_bar=True)

# --------------------------------------------------------- Decimating surfaces
W_322 = W_322.decimate(0.74)
W_422 = W_422.decimate(0.80)
W_332 = W_332.decimate(0.95)
W_333_0 = W_333_0.decimate(0.96)
W_333_1 = W_333_1.decimate(0.24)

# ---------------------------------------------------------- Flipped W surfaces
W_322_flip = W_322.reflect([1,0,0]).reflect([0,0,1])
W_422_flip = W_422.reflect([1,0,0]).reflect([0,0,1])
W_332_flip = W_332.reflect([1,0,0]).reflect([0,0,1])
W_333_0_flip = W_333_0.reflect([1,0,0]).reflect([0,0,1])
W_333_1_flip = W_333_1.reflect([1,0,0]).reflect([0,0,1])

# ----------------------------------------------------------------------- Plot

pv.global_theme.font.family = 'times'
pl = pv.Plotter()

pl.add_mesh(behavior_space, opacity=0.3)

pl.add_mesh(GHZ1, show_edges=True, color='blue')
pl.add_mesh(GHZ2, show_edges=True, color='blue')

pl.add_mesh(W_322, show_edges=True, color='green')
pl.add_mesh(W_332, show_edges=True, color='yellow')
pl.add_mesh(W_422, show_edges=True, color='purple')
pl.add_mesh(W_333_0, show_edges=True, color='red')
pl.add_mesh(W_333_1, show_edges=True, color='cyan')

pl.add_mesh(W_322_flip, show_edges=True, color='green')
pl.add_mesh(W_332_flip, show_edges=True, color='yellow')
pl.add_mesh(W_422_flip, show_edges=True, color='purple')
pl.add_mesh(W_333_0_flip, show_edges=True, color='red')
pl.add_mesh(W_333_1_flip, show_edges=True, color='cyan')

# # red cyan intersection
# pl.add_mesh(pv.Sphere(0.005, (1/3, -5/27, -5/9)), color='black')
# # yellow purple intersection
# pl.add_mesh(pv.Sphere(0.005, (1/3, 1/9, -5/9)), color='black')
# # green W-D1 line intersection
# pl.add_mesh(pv.Sphere(0.005, (0, 0, -1)), color='black')
# # green red intersection
# e3_intersection = np.roots([1, 6, 24, -6, -9])[3].real
# pl.add_mesh(pv.Sphere(0.005, (-e3_intersection/3,-1/3, e3_intersection)), color='black')
# # green red cyan yellow intersection
# pl.add_mesh(pv.Sphere(0.005, (-2+np.sqrt(5), 9-4*np.sqrt(5), 6-3*np.sqrt(5))), color='black')
# # cyan purple yellow W-D0 line intersection
# pl.add_mesh(pv.Sphere(0.005, (1/2, 0, -1/2)), color='black')

pl.add_point_labels([(0, 1, 0)], [r'$GHZ$'], show_points=True, point_size=1, 
                    point_color='black', font_family='times', always_visible=True)
pl.add_point_labels([(0, 0, 0)], [r'$U$'], show_points=True, point_size=10, 
                    point_color='black', font_family='times', always_visible=True, 
                    render_points_as_spheres=True)
pl.add_point_labels([(1, 1, 1)], [r'$D_+$'], show_points=True, point_size=1, 
                    point_color='black', font_family='times', always_visible=True)
pl.add_point_labels([(-1, 1, -1)], [r'$D_-$'], show_points=True, point_size=1, 
                    point_color='black', font_family='times', always_visible=True)
pl.add_point_labels([(1/3, -1/3, -1)], [r'$W$'], show_points=True, point_size=1, 
                    point_color='black', font_family='times', always_visible=True)
pl.add_point_labels([(-1/3, -1/3, 1)], [r'$\overline{W}$'], show_points=True, point_size=1,
                    point_color='black', font_family='times', always_visible=True)

marker = pv.create_axes_marker(xlabel=r'', ylabel=r'', zlabel=r'', 
                               line_width=2, cone_radius=0.2, shaft_length=0.8, 
                               tip_length=0.2)
pl.add_actor(marker)
pl.add_point_labels([(1, 0, 0)], [r'$E_1$'], show_points=False, 
                    point_color='black', font_family='times', 
                    shape=None, always_visible=True)
pl.add_point_labels([(0.1, 0.9, 0.1)], [r'$E_2$'], show_points=False, 
                    point_color='black', font_family='times', 
                    shape=None, always_visible=True)
pl.add_point_labels([(0, 0, 1)], [r'$E_3$'], show_points=False, 
                    point_color='black', font_family='times', 
                    shape=None, always_visible=True)

pl.camera_position = ((3.54, 0.46, 5.66),
                      (0, 0, 0),
                      (-0.74, -0.45, 0.5))
pl.camera_position = ((1.34, -1.39, 6.41),
                      (0, 0, 0),
                      (-0.11, 0.97, 0.23))
pl.show()

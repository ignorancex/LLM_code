import os
import numpy as np

from astropy import units as u
from astropy.table import Table


class KinematicDistance(object):
    def __init__(self, filename=None):
        self.galaxy_data = 'Reid+19'
        self.bdc_version = '2.4'
        self.verbose = False

    def initialize(self):
        self.rotation_curve_parameters()

    def rotation_curve_parameters(self):
        dirname = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__)))
        path_to_file = os.path.join(
            dirname, 'BDC', 'v' + self.bdc_version, 'galaxy_data_Univ.inp')

        gdU_list = []

        with open(path_to_file, 'r') as rFile:
            for line in rFile:
                line = line.partition('!')[0]
                line = line.rstrip()
                gdU_list.append(line)

        # Galactic structure parameters
        self.Ro = float(gdU_list[2])  # Ro [kpc]
        self.a1 = float(gdU_list[3])  # ~To [km/s]
        a2, a3 = gdU_list[4].split()  # Univ rotation curve parameters near 0.9, 1.5
        self.a2, self.a3 = float(a2), float(a3)

        # Solar Motion parameters
        self.Uo = float(gdU_list[5])  # Uo toward G.C. [km/s]
        self.Vo = float(gdU_list[6])  # Vo toward Gal. rotation [km/s]
        self.Wo = float(gdU_list[7])  # Wo toward North Gal.Pole [km/s]

        # Source peculiar motion in its local Galactocentric frame
        # (G.C.is as viewed by source, not Sun)
        self.Us = float(gdU_list[8])  # Us toward G.C. [km/s]
        self.Vs = float(gdU_list[9])  # Vs toward Gal. rotation [km/s]
        self.Ws = float(gdU_list[10])  # Ws toward North Gal.Pole [km/s]

        # Minimum/maximum Galactic longitude to consider (where models are valid)
        self.glong_min = float(gdU_list[11])
        self.glong_max = float(gdU_list[12])

        if self.verbose:
            print("""\
            Data from {}
            Galaxy model (Ro, a1, a2, a3): {}, {}, {}, {}
            Solar Motion (Uo, Vo, Wo): {}, {}, {}
            Source Peculiar Motion (Us, Vs, Ws): {}, {}, {}
            """. format(self.galaxy_data, self.Ro,
                        self.a1, self.a2, self.a3,
                        self.Uo, self.Vo, self.Wo,
                        self.Us, self.Vs, self.Ws))

    def Univ_RC(self, Rs):
        """Disk plus halo parameterization of rotation curve...

        see http://ned.ipac.caltech.edu/level5/March01/Battaner/node7.html
        referenced as following  Persic and Salucci (1997)

        Returns T(Rs) and T(Ro)
        """
        beta = 0.72  # 0.72 + 0.44 ln(L/L*)
        V2_Ropt = self.a1**2  # a1 = V(Ropt)
        # a2 = Ropt/Ro; (NB: Ropt=3.2Rd encloses 83% of light)
        # where Rd=scale length ~ 4.5 kpc (but often ~3 kpc used)
        # P.C. van der Kruit & K.C. Freeman ARAA, 2011, 49, 301
        Ropt = self.a2 * self.Ro
        a = self.a3  # "a" = 1.5(L/L*)^(1/5)

        # So, for L = L*, beta=0.72; a1~To; a2~3.2*(3.0to4.5)/8.3 = 1.2to1.7; a3~1.5

        for i_rho in [1, 2]:
            # Calculate T(Rs) and then T(Ro)...
            rho = Rs / Ropt
            if i_rho == 2:
                rho = self.Ro / Ropt

            # Exponential thin disk component...
            top = 1.97 * rho**1.22
            bot = (rho**2 + 0.78**2)**1.43
            V2_disk = V2_Ropt * beta * top / bot

            # Halo component...
            top = rho**2
            bot = rho**2 + a**2
            V2_halo = V2_Ropt*(1. - beta)*(1. + a**2)*top/bot

            # Total...
            Theta = np.sqrt(V2_disk + V2_halo)  # km/s

            if i_rho == 1:
                Tr = Theta
            else:
                To = Theta

        return Tr, To

    def Univ_RC_from_note(self, Rs):
        """Disk plus halo parameterization of rotation curve...

        see Persic, Salucci and Stel 1996 "Note Added in Proof"
        NB: doesn't use "a1" parameter
        """
        _lambda = (self.a3 / 1.5)**5  # L/L*

        # a2 = Ropt/Ro; (NB: Ropt=3.2Rd encloses 83% of light)
        # where Rd=scale length ~ 2.5 kpc if Ro ~ 8.1
        # P.C. van der Kruit & K.C. Freeman ARAA, 2011, 49, 301
        Ropt = self.a2 * self.Ro

        rho = Rs / Ropt

        #  Calculate Tr...
        log_lam = np.log10(_lambda)

        term1 = 200 * _lambda**0.41

        top = 0.75 * np.exp(-0.4 * _lambda)
        bot = 0.47 + 2.25 * _lambda**0.4
        term2 = np.sqrt(0.80 + 0.49*log_lam + (top/bot))

        top = 1.97 * rho**1.22
        bot = (rho**2 + 0.61)**1.43
        term3 = (0.72 + 0.44 * log_lam) * (top/bot)

        top = rho**2
        bot = rho**2 + 2.25 * _lambda**0.4
        term4 = 1.6 * np.exp(-0.4 * _lambda) * (top/bot)

        Tr = (term1/term2) * np.sqrt(term3 + term4)

        return Tr

    def kinematic_distance_Univ(self, v_proj, gal_long, Rs):
        """
        Calculate kinematic distance given Vlsr (projected in Gal plane)
        and information required to construct the kinematic model.
        Returns both near and far distances (when appropriate).
        Uses Persic's "Universal" rotation curve formulation.
        """
        import numpy as np

        glongrad = np.radians(gal_long)

        cos_l = np.cos(glongrad)
        sin_l = np.sin(glongrad)

        Rosinl = self.Ro * sin_l
        Rocosl = self.Ro * cos_l

        if self.bdc_version == '1.0':
            Tr, To = self.Univ_RC(Rs)
        elif self.bdc_version == '2.4':
            Tr = self.Univ_RC_from_note(Rs)
            #  Get To from rotation curve
            To = self.Univ_RC_from_note(self.Ro)

        Tosinl = To*sin_l
        Trsinl = Tr*sin_l

        # rootterm = Rocosl**2 + (Trsinl/(Tosinl/Ro + Vlsr/Ro))**2 - Ro**2
        rootterm = Rocosl**2 + (
            Trsinl / (Tosinl / self.Ro + v_proj / self.Ro))**2 - self.Ro**2

        if rootterm < 0:
            rootterm = 0.

        if ((gal_long >= 0.) and (gal_long < 90.)):
            D_near = Rocosl - np.sqrt(rootterm)
            D_far = Rocosl + np.sqrt(rootterm)

        if ((gal_long >= 90.) and (gal_long <= 270.)):
            D_near = Rocosl + np.sqrt(rootterm)
            D_far = D_near

        if ((gal_long > 270.) and (gal_long < 360.)):
            D_near = Rocosl - np.sqrt(rootterm)
            D_far = Rocosl + np.sqrt(rootterm)

        return D_near, D_far

    def calc_Dk_Univ(self, farnear, gal_long, gal_lat, v_lsr):
        """
        Calculate revised Vlsr by converting standard Vlsr back to
        heliocentric, apply modern Solar Motion values (Uo,Vo,Wo), and
        remove effects of average source non-ciruclar motion (Us,Vs,Ws).
        Then calculate kinematic distance using the Persic "Universal"
        rotation curve specified by Ro, a1, a2, and a3
        """
        import numpy as np

        n_iter_max = 100

        gal_long_rad = np.radians(gal_long)  # radians
        cos_l = np.cos(gal_long_rad)
        sin_l = np.sin(gal_long_rad)

        gal_lat_rad = np.radians(gal_lat)  # radians
        cos_b = np.cos(gal_lat_rad)
        sin_b = np.sin(gal_lat_rad)

        # Convert to true Heliocentric frame (v_rad)
        # Add back Sun's peculiar motion to radial velocity
        # Use old Standard Solar Motion (Sun moves at 20 km/s toward
        # 18h, +30deg in 1900 coordinates), since this has been used to
        # define V_lsr:
        Uo_IAU = 10.27  # km/s precessed to J2000
        Vo_IAU = 15.32
        Wo_IAU = 7.74

        v_helio = v_lsr - (Vo_IAU*sin_l + Uo_IAU*cos_l)*cos_b - Wo_IAU*sin_b

        # Make "new" V(LSR) using best Solar Motion
        v_newlsr = v_helio + (
            self.Vo * sin_l + self.Uo * cos_l)*cos_b + self.Wo * sin_b

        # Remove effects of common peculiar motions specified in
        # "galaxy_data_Univ.inp" file

        # In general, need to know distance to get source Galactocentric
        # radius (Rs) to evalute rotation curve.   So must iterate...
        n_iter = 0
        del_d = 99.
        Dk = 3.

        while ((del_d > 0.01) and (n_iter < n_iter_max)):
            # Save old value of kinematic distance
            Dk_old = Dk

            # Calculate "gamma" angle and projected Galactocentric radius
            d_proj = Dk*cos_b  # kpc in Gal Plane
            r_sq = self.Ro**2 + d_proj**2 - 2 * self.Ro * d_proj * cos_l
            r_proj = np.sqrt(r_sq)  # kpc in Gal Plane

            # Calculate Galactocentric longitude (beta in paper)...
            sin_beta = d_proj * sin_l / r_proj
            cos_beta = (self.Ro - d_proj * cos_l) / r_proj
            beta = np.arctan2(sin_beta, cos_beta)  # radians, CHECK if arctan2 == atan2 in fortran!!
            beta_deg = np.degrees(beta)  # deg

            # Calculate Sun-Maser-GC angle...
            gamma = np.pi - gal_long_rad - beta  # radians
            cos_gamma = np.cos(gamma)
            sin_gamma = np.sin(gamma)

            v_fixed = v_newlsr - (self.Vs * sin_gamma - self.Us * cos_gamma) * cos_b - self.Ws * sin_b  # km/s

            # Calculate a kinematic distance using best Ro, To and dTdr

            Rs = r_proj
            v_proj = v_fixed*cos_b
            D_near, D_far = self.kinematic_distance_Univ(
                v_proj, gal_long, Rs)

            Dk = D_near

            if (farnear != 0.):
                Dk = D_far

            # Ignore "farnear" flag if one of the values is zero
            if ((D_near <= 0.) and (D_far > 0.)):
                Dk = D_far
            if ((D_far <= 0.) and (D_near > 0.)):
                Dk = D_near

            del_d = abs(Dk - Dk_old)
            n_iter = n_iter + 1

        v_lsr_rev = v_fixed

        return Dk

    def calc_tangent_point_velocity(self, gal_long, gal_lat, Dk):
        """
        Reverse engineer vlsr for specified kinematic distance of the tangent
        point.
        """
        import numpy as np

        gal_long_rad = np.radians(gal_long)  # radians
        cos_l = np.cos(gal_long_rad)
        sin_l = np.sin(gal_long_rad)

        gal_lat_rad = np.radians(gal_lat)  # radians
        cos_b = np.cos(gal_lat_rad)
        sin_b = np.sin(gal_lat_rad)

        # Convert to true Heliocentric frame (v_rad)
        # Add back Sun's peculiar motion to radial velocity
        # Use old Standard Solar Motion (Sun moves at 20 km/s toward
        # 18h, +30deg in 1900 coordinates), since this has been used to
        # define V_lsr:
        Uo_IAU = 10.27  # km/s precessed to J2000
        Vo_IAU = 15.32
        Wo_IAU = 7.74

        # Calculate "gamma" angle and projected Galactocentric radius
        d_proj = Dk*cos_b  # kpc in Gal Plane
        r_sq = self.Ro**2 + d_proj**2 - 2 * self.Ro * d_proj * cos_l
        r_proj = np.sqrt(r_sq)  # kpc in Gal Plane

        # Calculate Galactocentric longitude (beta in paper)...
        sin_beta = d_proj * sin_l / r_proj
        cos_beta = (self.Ro - d_proj * cos_l) / r_proj
        beta = np.arctan2(sin_beta, cos_beta)  # radians, CHECK if arctan2 == atan2 in fortran!!
        beta_deg = np.degrees(beta)  # deg

        # Calculate Sun-Maser-GC angle...
        gamma = np.pi - gal_long_rad - beta  # radians
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)

        #  TODO: modify the following to also reverse engineer distances
        #  other than the tangent point distance?

        Rs = r_proj

        glongrad = np.radians(gal_long)

        cos_l = np.cos(glongrad)
        sin_l = np.sin(glongrad)

        Rosinl = self.Ro * sin_l
        Rocosl = self.Ro * cos_l

        if self.bdc_version == '1.0':
            Tr, To = self.Univ_RC(Rs)
        elif self.bdc_version == '2.4':
            Tr = self.Univ_RC_from_note(Rs)
            #  Get To from rotation curve
            To = self.Univ_RC_from_note(self.Ro)

        Tosinl = To*sin_l
        Trsinl = Tr*sin_l

        # rootterm = Rocosl**2 + (Trsinl/(Tosinl/Ro + Vlsr/Ro))**2 - Ro**2
        term = (Trsinl / (np.sqrt(self.Ro**2 - Rocosl**2)))
        v_proj = term * self.Ro - Tosinl

        v_fixed = v_proj / cos_b

        v_newlsr = v_fixed + (self.Vs * sin_gamma - self.Us * cos_gamma) * cos_b - self.Ws * sin_b  # km/s

        # Make "new" V(LSR) using best Solar Motion
        v_helio = v_newlsr - (
            self.Vo * sin_l + self.Uo * cos_l)*cos_b + self.Wo * sin_b

        v_lsr = v_helio + (Vo_IAU*sin_l + Uo_IAU*cos_l)*cos_b - Wo_IAU*sin_b

        return v_lsr

    # Get "standard" kinematic distance(s) for informational use only.
    # (Later Prob_KD will be directly calculated from velocity differences,
    # avoiding complications associated with allowing for velocity uncertainties.)

    def calc_kinematic_distance(self, gal_long, gal_lat, v_lsr):
        """"""
        farnear_flag = 0.  # near distance flag

        Dk_near = self.calc_Dk_Univ(farnear_flag, gal_long, gal_lat, v_lsr)

        if self.verbose:
            if (Dk_near > 0.):
                print('Kinematic distance (near): {:7.2f}'.format(Dk_near))

        farnear_flag = 1.  # far distance flag
        Dk_far = self.calc_Dk_Univ(farnear_flag, gal_long, gal_lat, v_lsr)

        if self.verbose:
            if (Dk_far > Dk_near):
                print('Kinematic distance (far): {:7.2f}'.format(Dk_far))

        # Check for pathological values...
        if ((Dk_near <= 0.) and (Dk_far <= 0.)):
            if self.verbose:
                print("unlikely V(LSR), longitude pair")

        return Dk_near, Dk_far

    def get_vmax(self, ell):
        """return vmax

        Calculates the maximum line of sight velocity for a given longitude
        for a Persic "Universal" rotation curve.  This may differ from the
        "tangent point" velocity.
        """
        V_max = 0

        # Only valid for quadrants 1 and 4 (and avoid edges)
        if (0. < ell < 90) or (270. < ell < 360):
            ell_rad = np.deg2rad(ell)
            sin_l = np.sin(ell_rad)
            cos_l = np.cos(ell_rad)
        else:
            #  TODO: implement warning
            print('Vmax calculation only valid for Quadrant 1 and 4')
            pass

        #  Walk outward from Sun and find maximum V
        #  (positive in Q1; negative in Q4)
        for d in range(1, 101):
            d *= 0.1
            y = self.Ro - d * cos_l  # kpc
            x = d * sin_l
            Rs = np.sqrt(x**2 + y**2)

            sin_beta = x / Rs
            cos_beta = y / Rs
            beta = np.arctan2(sin_beta, cos_beta)  # rad
            ellbeta = ell_rad + beta  # rad

            Tr = self.Univ_RC_from_note(Rs)
            #  Get To from rotation curve
            To = self.Univ_RC_from_note(self.Ro)

            V_los = Tr * np.sin(ellbeta) - To*sin_l  # km/s

            if abs(V_los) > abs(V_max):
                V_max = V_los

        return V_max


def compare_distances(vlsr, vel_tp, dist, dist_kd_n, dist_kd_f, dist_kd_t,
                      e_dist=None, e_dist_factor=0.2, e_dist_min=1,
                      vel_factor=10):
    if e_dist is None:
        e_dist = max(e_dist_min, e_dist_factor * dist)

    diff_kd_n = abs(dist - dist_kd_n)
    diff_kd_f = abs(dist - dist_kd_f)
    diff_kd_t = abs(dist - dist_kd_t)

    if (abs(vlsr - vel_tp) < vel_factor) or (vlsr > vel_tp):
        return 'T', dist_kd_t

    if dist < dist_kd_n:
        return 'N', dist_kd_n
    elif dist > dist_kd_f:
        return 'F', dist_kd_f
    elif diff_kd_t < min(diff_kd_n, diff_kd_f):
        return 'U', dist_kd_t
    elif diff_kd_n < diff_kd_f:
        return 'N', dist_kd_n
    else:
        return 'F', dist_kd_f


def infer_galactocentric_radius(glon, dist, R_0=8.5):
    glon_rad = np.radians(glon)
    term_1 = (dist - R_0*np.cos(glon_rad))**2
    term_2 = (R_0 * np.sin(glon_rad))**2
    return np.sqrt(term_1 + term_2)


def calculate_kinematic_distances(glon, rgal, R_0=8.5):
    glon_rad = np.radians(glon)
    root = np.sqrt(rgal**2 - (R_0 * np.sin(glon_rad))**2)
    term = R_0 * np.cos(glon_rad)
    d_n = term - root
    d_f = term + root
    return d_n, d_f


def infer_kda_solution(dist, dist_n, dist_f, threshold=1.):
    kda = []
    for d, d_n, d_f in zip(dist, dist_n, dist_f):
        if abs(d_n - d_f) < threshold:
            kda.append('T')
        elif abs(d_n - d) < abs(d_f - d):
            kda.append('N')
        else:
            kda.append('F')
    return kda


def infer_kinematic_distances(glon, dist, R_0=None, threshold=1., glat=None):
    if glat is not None:
        glat_rad = np.radians(glat)
        dist = np.cos(glat_rad) * dist
    rgal = infer_galactocentric_radius(glon, dist, R_0=R_0)
    dist_n, dist_f = calculate_kinematic_distances(glon, rgal, R_0=R_0)
    kda = infer_kda_solution(dist, dist_n, dist_f, threshold=threshold)
    return dist_n, dist_f, kda

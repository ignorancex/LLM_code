"""Parameterizations for form factors."""

from math import pi, sqrt

import numpy as np
from scipy.special import kn

# Physical constants:
M = 0.938272088  # Proton mass [GeV]
kappa = 1.7928473446  # Anomalous magnetic moment of the proton
hbarc = 0.1973269804  # hbar times c [GeV*fm]


def get_radius(b2, b2_sigma):
    """Calculate the proton radius and its statistical uncertainty."""
    radius = np.sqrt(1.5 * (b2 + kappa / (M * M)))
    radius_stat = 0.75 * b2_sigma / radius
    return radius * hbarc, radius_stat * hbarc


def get_b2(params, cov):
    """Calculate squared transverse charge radius and its uncertainty."""
    L = params[0]
    if len(params) == 1:
        b2 = 8 / L ** 2
        b2_sigma = 16 * np.sqrt(cov[0, 0]) / L ** 3
    else:
        a1 = params[1]
        b2 = 8 * (1 + sqrt(2) * a1) / L ** 2
        var_L = 4 * (1 + sqrt(2) * a1) ** 2 * cov[0, 0] / L ** 2
        var_cross = -4 * sqrt(2) * (1 + sqrt(2) * a1) * cov[0, 1] / L
        b2_sigma = 8 * sqrt(var_L + 2 * cov[1, 1] + var_cross) / L ** 2
    return b2, b2_sigma


def dipole_ffs(Q2):
    """Calculate dipole form factors."""
    GE = 1 / (1 + Q2 / 0.71) ** 2
    GM = (1 + kappa) * GE
    return GE, GM


def calc_cs(E, Q2, fit_params, order):
    """Calculate model cross section over dipole cross section."""

    # Kinematic parameters:
    tau = Q2 / (4 * M * M)
    theta = np.arccos(1 - M * Q2 / (E * (2 * M * E - Q2)))
    epsilon = 1.0 / (1 + 2 * (1 + tau) * np.tan(0.5 * theta) ** 2)

    # Calculate model form factors:
    F1, F2 = calc_ffs(Q2, fit_params, order)
    GE = F1 - tau * kappa * F2
    GM = F1 + kappa * F2

    # Calculate dipole form factors:
    GE_dip, GM_dip = dipole_ffs(Q2)

    # Calculate reduced cross sections:
    cs = epsilon * GE ** 2 + tau * GM ** 2
    cs_dip = epsilon * GE_dip ** 2 + tau * GM_dip ** 2

    # Return cross section ratio:
    return cs / cs_dip


def calc_ge_gm(Q2_range, params, order):
    """Calculate GE and GM over dipole form factors."""
    F1, F2 = calc_ffs(Q2_range, params, order)
    GE = F1 - Q2_range * kappa * F2 / (4 * M * M)
    GM = F1 + kappa * F2
    GE_dip, GM_dip = dipole_ffs(Q2_range)
    return GE / GE_dip, GM / GM_dip


def get_init(order):
    """Return initial guess for fit parameters."""
    lambda_guess = [1.0]
    coeff_guess = [0.0] * (2 * order)
    return lambda_guess + coeff_guess


def calc_ffs(Q2, fit_params, order):
    """Calculate F1 and F2 using expansion to the given order."""

    L = fit_params[0]
    y = Q2 / L ** 2
    d2 = (1 + y) * (1 + y)
    d3 = d2 * (1 + y)
    F1 = 1 / d2
    F2 = 1 / d3

    if order >= 1:
        a1, b1 = fit_params[1], fit_params[2]
        d4 = d3 * (1 + y)
        d5 = d4 * (1 + y)
        F1 -= a1 * y * (y + 4) / (sqrt(2) * d4)
        F2 -= b1 * sqrt(3) * y * (y + 5) / (sqrt(5) * d5)
    if order >= 2:
        a2, b2 = fit_params[3], fit_params[4]
        y2 = y * y
        d6 = d5 * (1 + y)
        d7 = d6 * (1 + y)
        F1 += a2 * y2 * (3 * y2 + 22 * y + 39) / (sqrt(26) * d6)
        F2 += b2 * y2 * (7 * y2 + 64 * y + 132) / (sqrt(110) * d7)
    if order >= 3:
        a3, b3 = fit_params[5], fit_params[6]
        y3 = y2 * y
        d8 = d7 * (1 + y)
        d9 = d8 * (1 + y)
        F1 -= a3 * y3 * (34 * y3 + 352 * y2 + 1187 * y + 1324) / (sqrt(4303) * d8)
        F2 -= b3 * y3 * (25 * y3 + 321 * y2 + 1260 * y + 1580) / (sqrt(1738) * d9)
    if order >= 4:
        a4, b4 = fit_params[7], fit_params[8]
        y4 = y3 * y
        d10 = d9 * (1 + y)
        d11 = d10 * (1 + y)
        F1 += a4 * y4 * (1635 * y4 + 21560 * y3 + 104885 * y2 + 225774 * y + 182520) / (78 * sqrt(1986) * d10)
        F2 += (
            b4
            * sqrt(3)
            * y4
            * (635 * y4 + 10324 * y3 + 58410 * y2 + 141476 * y + 125935)
            / (sqrt(3979546) * d11)
        )
    if order >= 5:
        a5, b5 = fit_params[9], fit_params[10]
        y5 = y4 * y
        d12 = d11 * (1 + y)
        d13 = d12 * (1 + y)
        F1 -= (
            a5
            * y5
            * (13257 * y5 + 210730 * y4 + 1324320 * y3 + 4152432 * y2 + 6528139 * y + 4127178)
            / (26 * sqrt(1375726) * d12)
        )
        F2 -= (
            b5
            * y5
            * (378679 * y5 + 7377979 * y4 + 53840262 * y3 + 189977062 * y2 + 329168959 * y + 225929067)
            / (sqrt(541950039098) * d13)
        )
    if order >= 6:
        a6, b6 = fit_params[11], fit_params[12]
        y6 = y5 * y
        d14 = d13 * (1 + y)
        d15 = d14 * (1 + y)
        F1 += (
            a6
            * y6
            * (
                374028837 * y6
                + 6925266446 * y5
                + 52983708001 * y4
                + 216093302432 * y3
                + 497585924843 * y2
                + 614700189538 * y
                + 318645165251
            )
            / (3 * sqrt(93936093987877977) * d14)
        )
        F2 += (
            b6
            * y6
            * (
                52471398 * y6
                + 1184074416 * y5
                + 10499540352 * y4
                + 48172744400 * y3
                + 122305611645 * y2
                + 164178056304 * y
                + 91446537252
            )
            / (sqrt(11712262381930331) * d15)
        )
    if order >= 7:
        a7, b7 = fit_params[13], fit_params[14]
        y7 = y6 * y
        d16 = d15 * (1 + y)
        d17 = d16 * (1 + y)
        F1 -= (
            a7
            * y7
            * (
                1930123093508 * y7
                + 40661785878056 * y6
                + 364995142413258 * y5
                + 1821579544284472 * y4
                + 5478404821372252 * y3
                + 9948436593025344 * y2
                + 10110352471868063 * y
                + 4437930583900664
            )
            / (sqrt(25252234362133821436582619) * d16)
        )
        F2 -= (
            b7
            * sqrt(3)
            * y7
            * (
                31938116286 * y7
                + 815655077286 * y6
                + 8468426106840 * y5
                + 47513556718256 * y4
                + 157574046514441 * y3
                + 311094988545093 * y2
                + 339991124114256 * y
                + 159095481965508
            )
            / (sqrt(14433264799785437997127) * d17)
        )
    if order >= 8:
        a8, b8 = fit_params[15], fit_params[16]
        y8 = y7 * y
        d18 = d17 * (1 + y)
        d19 = d18 * (1 + y)
        F1 += (
            a8
            * y8
            * (
                50465305304713197 * y8
                + 1189218662755495176 * y7
                + 12212927815630214826 * y6
                + 71793235471158539100 * y5
                + 265053242530285299174 * y4
                + 630413523565947135360 * y3
                + 944190467383460755491 * y2
                + 814513841411182730558 * y
                + 309888477638191060236
            )
            / (2 * sqrt(4775220668433837190423830939806933) * d18)
        ).astype("float64")
        F2 += (
            b8
            * y8
            * (
                798068857505835 * y8
                + 22683688182424128 * y7
                + 268872520438683960 * y6
                + 1775435313228384800 * y5
                + 7227013034952223085 * y4
                + 18693288089035399920 * y3
                + 30125114916594504550 * y2
                + 27724800951475092280 * y
                + 11174551176065608620
            )
            / (2 * sqrt(823065094955737151785833271981) * d19)
        ).astype("float64")

    return F1, F2


def rho_2pole(Lambda, b):
    """Calculate the 2-pole transverse charge density."""
    with np.errstate(invalid="ignore"):
        return (Lambda ** 2) / (4 * pi) * np.where(b > 0, Lambda * b * kn(1, Lambda * b), 1)


def rho_3pole(Lambda, b):
    """Calculate the 3-pole transverse charge density."""
    with np.errstate(invalid="ignore"):
        return (Lambda ** 2) / (16 * pi) * np.where(b > 0, ((Lambda * b) ** 2) * kn(2, Lambda * b), 2)


def calc_rho(b, fit_params, order):
    """Calculate rho1 and rho2 using expansion to the given order."""

    Lambda = fit_params[0]
    x = (Lambda * b / 2) ** 2
    sum1, sum2 = 1, 1

    if order >= 1:
        a1, b1 = fit_params[1], fit_params[2]
        sum1 += a1 * (x - 2) / (2 * sqrt(2))
        sum2 += b1 * (x - 3) / sqrt(15)

    if order >= 2:
        a2, b2 = fit_params[3], fit_params[4]
        x2 = x * x
        sum1 += a2 * (x2 - 15 * x + 18) / (6 * sqrt(26))
        sum2 += b2 * (5 * x2 - 96 * x + 168) / (24 * sqrt(110))

    if order >= 3:
        a3, b3 = fit_params[5], fit_params[6]
        x3 = x2 * x
        sum1 += a3 * (13 * x3 - 636 * x2 + 5328 * x - 4896) / (144 * sqrt(4303))
        sum2 += b3 * (11 * x3 - 645 * x2 + 6840 * x - 9000) / (360 * sqrt(1738))

    if order >= 4:
        a4, b4 = fit_params[7], fit_params[8]
        x4 = x3 * x
        sum1 += a4 * (331 * x4 - 37620 * x3 + 997200 * x2 - 6105600 * x + 4708800) / (224640 * sqrt(1986))
        sum2 += b4 * (79 * x4 - 10368 * x3 + 327420 * x2 - 2517120 * x + 2743200) / (1440 * sqrt(11938638))

    if order >= 5:
        a5, b5 = fit_params[9], fit_params[10]
        x5 = x4 * x
        sum1 += (
            a5
            * (676 * x5 - 148095 * x4 + 8964900 * x3 - 169866000 * x2 + 844128000 * x - 572702400)
            / (1123200 * sqrt(1375726))
        )
        sum2 += (
            b5
            * (25187 * x5 - 6219045 * x4 + 433086780 * x3 - 9730620900 * x2 + 60331975200 * x - 57256264800)
            / (151200 * sqrt(541950039098))
        )

    if order >= 6:
        a6, b6 = fit_params[11], fit_params[12]
        x6 = x5 * x
        sum1 += (
            a6
            * (
                2063589 * x6
                - 773963946 * x5
                + 89315190020 * x4
                - 3813443600400 * x3
                + 57647082528000 * x2
                - 245641576521600 * x
                + 150808427078400
            )
            / (1209600 * sqrt(93936093987877977))
        )
        sum2 += (
            b6
            * (
                10758527 * x6
                - 4470700992 * x5
                + 579988357200 * x4
                - 28400417095680 * x3
                + 507142207353600 * x2
                - 2682494309376000 * x
                + 2284898508748800
            )
            / (43545600 * sqrt(11712262381930331))
        )

    if order >= 7:
        a7, b7 = fit_params[13], fit_params[14]
        x7 = x6 * x
        sum1 += (
            a7
            * (
                45520737893 * x7
                - 26924324867832 * x6
                + 5278224631430016 * x5
                - 425759749642523520 * x4
                + 14346687224555078400 * x3
                - 183417735939554995200 * x2
                + 692958826702575820800 * x
                - 392225718176422502400
            )
            / (203212800 * sqrt(25252234362133821436582619))
        ).astype("float64")
        sum2 += (
            b7
            * (
                1088649253 * x7
                - 704386668111 * x6
                + 152711233046064 * x5
                - 13818114573011280 * x4
                + 532637816505903360 * x3
                - 8018272971174009600 * x2
                + 37444137784326604800 * x
                - 29206053167416473600
            )
            / (304819200 * sqrt(43299794399356313991381))
        ).astype("float64")

    if order >= 8:
        a8, b8 = fit_params[15], fit_params[16]
        x8 = x7 * x
        sum1 += (
            a8
            * (
                554741322987583 * x8
                - 487172294964340488 * x7
                + 149751895269168650304 * x6
                - 20387420658363960649728 * x5
                + 1288140367546428359362560 * x4
                - 36380562241241713853030400 * x3
                + 407703880727121822236467200 * x2
                - 1396463730652618897541529600 * x
                + 738374111555444781071155200
            )
            / (29262643200 * sqrt(4775220668433837190423830939806933))
        ).astype("float64")
        sum2 += (
            b8
            * (
                13257956830459 * x8
                - 12611371036263360 * x7
                + 4234744375323283920 * x6
                - 636558769525100083200 * x5
                + 45033530323713182496000 * x4
                - 1451720205407331591782400 * x3
                + 19104497240582360936448000 * x2
                - 80590835034093195657216000 * x
                + 58384010565562228807680000
            )
            / (146313216000 * sqrt(823065094955737151785833271981))
        ).astype("float64")

    rho1 = rho_2pole(Lambda, b) * sum1
    rho2 = rho_3pole(Lambda, b) * sum2

    return rho1, rho2

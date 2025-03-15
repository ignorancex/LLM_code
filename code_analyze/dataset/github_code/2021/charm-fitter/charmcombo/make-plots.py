#!/usr/bin/env python3
'''
Produce 1D and 2D plots for a single combination, or comparing different
combinations. The prob method is employed.

Examples:
- plot the world average from 2024:
    `python3 make-plots.py --1d --2d -c 50 --param theoretical`
- compare the world averages from 2024, with and without allowing for CP violation in doubly Cabibbo-suppressed decays:
    `python3 make-plots.py --lhcb -c 50 50 --title "WA 2024" "WA 2024, A_{CP}(K#pi) = 0" --no-dcs-cpv 0 1 --1d --2d -a`
'''

import argparse
from copy import copy
import subprocess


_not_phen_params = ['x12', 'y12', 'phiG', 'phiM']
_not_supw_params = ['x', 'y', 'qop', 'phi', 'phiG', 'Acp_KK', 'Acp_PP', 'Acp_KP', 'cot_delta_KK', 'cot_delta_PP']
_not_theo_params = ['x', 'y', 'qop', 'phi']
_full_fsc_params = ['cot_delta_KK', 'cot_delta_PP']
_cpv_decay_params = ['Acp_KK', 'Acp_PP', 'cot_delta_KK', 'cot_delta_PP']
_full_fsc_combinations = [42]


class Scan1d:
    def __init__(self, var, x_min=None, x_max=None, npoints=200):
        self.var = var
        self.x_min = x_min
        self.x_max = x_max
        self.npoints = npoints
        self.phenomenological = var not in _not_phen_params
        self.superweak = var not in _not_supw_params
        self.theoretical = var not in _not_theo_params
        self.full_fsc_only = var in _full_fsc_params


class Scan2d:
    def __init__(self, varx, vary, x_min=None, x_max=None, y_min=None, y_max=None, npoints_x=80, npoints_y=80):
        self.id = f'{varx}_{vary}'
        self.var_x = varx
        self.var_y = vary
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.npoints_x = npoints_x
        self.npoints_y = npoints_y
        self.phenomenological = varx not in _not_phen_params and vary not in _not_phen_params
        self.superweak = varx not in _not_supw_params and vary not in _not_supw_params
        self.theoretical = varx not in _not_theo_params and vary not in _not_theo_params
        self.full_fsc_only = varx in _full_fsc_params or vary in _full_fsc_params


def get_scans_1d(combinations, scan_dy_rs=False):
    before_2022 = 500 in combinations or any(n < 30 for n in combinations)
    if before_2022:
        scans = [
            Scan1d('x12',           0. ,   0.8 ),
            Scan1d('y12',           0.4,   0.9 ),
            Scan1d('phiM',         -0.3,   0.3  ),
            Scan1d('phiG',         -0.4,   0.4),
            Scan1d('qop',          -0.2  , 0.2  ),
            Scan1d('phi',          -0.4  , 0.4  ),
            Scan1d('x',             0. ,   0.8 ),
            Scan1d('y',             0.4,   0.9 ),
            Scan1d('Delta_Kpi',    -0.8,   0.5 ),
            Scan1d('R_Kpi',         0.325 , 0.36 ),
            Scan1d('Acp_KK',       -0.5,   0.5 ),
            Scan1d('Acp_PP',       -0.4,   0.7 ),
            Scan1d('cot_delta_KK', -4e2,   4e2 ),
            Scan1d('cot_delta_PP', -4e2,   4e2 ),
            ]
    else:
        scans = [
            Scan1d('x12',           0.2,   0.7 ),
            Scan1d('y12',           0.5,   0.8 ),
            Scan1d('phiM',         -0.125,  0.125),
            Scan1d('phiG',         -0.15,  0.15),
            Scan1d('qop',          -0.075, 0.075),
            Scan1d('phi',          -0.125, 0.125),
            Scan1d('x',             0.2,   0.7 ),
            Scan1d('y',             0.5,   0.8 ),
            Scan1d('Delta_Kpi',    -0.5,   0.2 ),
            Scan1d('R_Kpi',         0.335, 0.355),
            Scan1d('Acp_KK',       -0.7,   0.7 ),
            Scan1d('Acp_PP',       -0.4,   0.7 ),
            Scan1d('Acp_KP', -2, 2),
            Scan1d('cot_delta_KK', -4e2,   4e2 ),
            Scan1d('cot_delta_PP', -4e2,   4e2 ),
            ]
    scans.extend([
        Scan1d('Delta_Kpipi0', -2., 2.),
        Scan1d('Delta_K3pi', -0.5, 1.5),
        Scan1d('k_K3pi', 0., 1.),
        Scan1d('k_Kpipi0', 0., 1.),
        Scan1d('r_K3pi', 4., 6.),
        Scan1d('r_Kpipi0', 4., 6.),
        Scan1d('F_pipipi0', 0.9, 1.),
    ])
    if scan_dy_rs:
        scans.append(Scan1d('DY_RS', 0., 0.4))
    return scans


def get_scans_2d(combinations, plot_acp_kp=True):
    before_2022 = 500 in combinations or any(n < 30 for n in combinations)
    if before_2022:
        scans = [
            Scan2d('x12', 'y12', 0., 0.8, 0.4, 0.9),
            Scan2d('phiM', 'phiG', -0.3, 0.3, -0.4, 0.4),
            Scan2d('Delta_Kpi', 'R_Kpi', -0.6, 0.4, 0.325, 0.36),
            Scan2d('x', 'y', 0., 0.8, 0.4, 0.9),
            Scan2d('qop', 'phi', -0.2, 0.2, -0.4, 0.4),
            Scan2d('Acp_KK', 'Acp_PP', -0.5, 0.5, -0.5, 0.7),
            ]
    else:
        scans = [
            Scan2d('x12', 'y12', 0.2, 0.6, 0.55, 0.75),
            Scan2d('phiM', 'phiG', -0.15, 0.15, -0.15, 0.15),
            Scan2d('Delta_Kpi', 'R_Kpi', -0.5, 0.05, 0.3375, 0.3525),
            Scan2d('x', 'y', 0.2, 0.6, 0.55, 0.75),
            Scan2d('qop', 'phi', -0.075, 0.075, -0.125, 0.075),
            Scan2d('Acp_KK', 'Acp_PP', -0.3, 0.3, -0.3, 0.5),
            Scan2d('Acp_KK', 'phiM', -0.3, 0.3, -0.15, 0.15),
            ]
    if plot_acp_kp:
        scans.extend([
            Scan2d('Acp_KK', 'Acp_KP', -0.3, 0.3, -2, 2),
            Scan2d('Acp_KP', 'phiM', -2, 2, -0.15, 0.15),
            Scan2d('Acp_KP', 'phiG', -2, 2, -0.15, 0.15),
            ])
    return scans


def get_base_options(param, combinations, titles, parfile, already_plotted, lhcb, no_dcs_cpv):
    options = [f'--param {param}', '--ps 1', '--pulls', '--pr', '--qh 29'] # , '--group off']
    for i in range(len(combinations)):
        fix_acp_kpi = no_dcs_cpv is not None and no_dcs_cpv[i]
        options.append('-c {}'.format(int(combinations[i]) + 1000 if fix_acp_kpi else combinations[i]))
        options.append('' if titles is None else f' --title "{titles[i]}"')
        if no_dcs_cpv is not None:
            options.append(f'--fix {"Acp_KP=0" if fix_acp_kpi else "none"}')
        if parfile is not None:
            options.append(f'--parfile {parfile}')
    if len(combinations) == 1:
        options.append('--leg off')
    if already_plotted:
        options.append('-a plot')
    if lhcb:
        options.append('--group "LHCb preliminary"')
    return options


def get_options_1d(param, combinations, titles, parfile, already_plotted, lhcb, no_dcs_cpv):
    return get_base_options(param, combinations, titles, parfile, already_plotted, lhcb, no_dcs_cpv)


def get_options_2d(param, combinations, titles, parfile, already_plotted, lhcb, no_dcs_cpv, n_contours=2):
    options = get_base_options(param, combinations, titles, parfile, already_plotted, lhcb, no_dcs_cpv)
    options.extend(['--magnetic', '--2dcl 1', f'--ncontours {n_contours}', '--ps 2'])
    if len(combinations) > 1:
        options.append('--leg 0.18:0.25')
    return options


def execute_cmd(command):
    print(f'Execute `{command}`')
    subprocess.run(command, shell=True)


def plot_1d(param, combinations, titles, parfile, already_plotted, scan_dy_rs, lhcb, no_dcs_cpv):
    plot_acp_kp = no_dcs_cpv is None or True not in no_dcs_cpv
    for scan in get_scans_1d([int(combo) for combo in combinations]):
        cc = copy(combinations)
        tt = copy(titles)
        i = 0
        while i < len(cc):
            if (scan.full_fsc_only and cc[i] not in _full_fsc_combinations) or (scan.var == 'Acp_KP' and plot_acp_kp and int(cc[i]) >= 1000):
            # or (scan.var in _cpv_decay_params and ((not isinstance(cc[i], int)) or cc[i] < 40))):  # TODO
                cc.pop(i)
                if tt is not None:
                    tt.pop(i)
            else:
                i += 1
        if len(cc) == 0 or not getattr(scan, param):
            continue
        options = get_options_1d(param, cc, tt, parfile, already_plotted, lhcb, no_dcs_cpv)
        options.append(f'--var {scan.var}')
        options.append(f'--npoints {scan.npoints}')
        if scan.x_min is not None and scan.x_max is not None:
            options.append(f'--scanrange {scan.x_min}:{scan.x_max}')
        if scan.var == 'DY_RS':
            if scan_dy_rs:
                options.append('--teststat 2')  # one-sided CL
            else:
                continue
        execute_cmd(f'bin/charmcombo {" ".join(options)}')


def plot_2d(param, combinations, titles, parfile, already_plotted, lhcb, no_dcs_cpv):
    plot_acp_kp = no_dcs_cpv is None or True not in no_dcs_cpv
    for scan in get_scans_2d([int(combo) for combo in combinations], plot_acp_kp=plot_acp_kp):
        cc = copy(combinations)
        tt = copy(titles)
        i = 0
        while i < len(cc):
            if (scan.full_fsc_only and cc[i] not in _full_fsc_combinations) or (any(v == 'Acp_KP' for v in [scan.var_x, scan.var_y]) and plot_acp_kp and int(cc[i]) >= 1000):
            # or (any(v in _cpv_decay_params for v in [scan.var_x, scan.var_y]) and ((not isinstance(cc[i], int)) or cc[i] < 40))):  # TODO
                cc.pop(i)
                if tt is not None:
                    tt.pop(i)
            else:
                i += 1
        if len(cc) == 0 or not getattr(scan, param):
            continue
        options = get_options_2d(param, cc, tt, parfile, already_plotted, lhcb, no_dcs_cpv)
        options.append(f'--var {scan.var_x}')
        options.append(f'--var {scan.var_y}')
        options.append(f'--npoints2dx {scan.npoints_x}')
        options.append(f'--npoints2dy {scan.npoints_y}')
        if scan.x_min is not None and scan.x_max is not None:
            options.append(f'--scanrange {scan.x_min}:{scan.x_max}')
        if scan.y_min is not None and scan.y_max is not None:
            options.append(f'--scanrangey {scan.y_min}:{scan.y_max}')
        execute_cmd(f'bin/charmcombo {" ".join(options)}')


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param', choices=['phenomenological', 'theoretical', 'superweak'],
                        default='theoretical', help='Parametrisation to be used in the combination')
    parser.add_argument('-c', '--combinations', nargs='+', default=[50], help='Combinations to plot')
    parser.add_argument('-t', '--titles', nargs='+', default=None,
                        help='Titles for the combinations to plot. If none, the default values are used.')
    parser.add_argument('--1d', action='store_true', dest='plot_1d', help='Make the 1d plots')
    parser.add_argument('--2d', action='store_true', dest='plot_2d', help='Make the 2d plots')
    parser.add_argument('--parfile', default=None, help='File with initial value of the parameters')
    parser.add_argument(
            '-a', '--already-plotted', action='store_true',
            help='Do not rerun the combination and just plot the results retrieving them from a previous execution.')
    parser.add_argument('--scan-dy-rs', action='store_true', help='Provide predictions for the value of DY(K- pi+).')
    parser.add_argument('--lhcb', action='store_true', help='Add the `LHCb` label to the plot')
    parser.add_argument('--no-dcs-cpv', type=str_to_bool, nargs='+', default=None, help='Set A_CP(K+ pi-) to zero')
    args = parser.parse_args()
    if args.titles is not None:
        assert len(args.titles) == len(args.combinations), 'You should provide one title for each combination'
    if args.no_dcs_cpv is not None:
        assert len(args.no_dcs_cpv) == len(args.combinations), 'You should tell whether CPV is allowed in DCS decays for each combination'
    fn_args = copy(vars(args))
    del fn_args['plot_1d'], fn_args['plot_2d']
    if args.plot_1d:
        plot_1d(**fn_args)
    del fn_args['scan_dy_rs']
    if args.plot_2d:
        plot_2d(**fn_args)


if __name__ == '__main__':
    parse_args()

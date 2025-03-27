import matplotlib as mpl


def mplchange():
    mpl.rcParams.update({'font.size': 30})
    mpl.rcParams['xtick.major.size'] = 12
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['xtick.minor.size'] = 8
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['xtick.minor.width'] = 1
    mpl.rcParams['ytick.major.size'] = 12
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['ytick.minor.size'] = 8
    mpl.rcParams['ytick.minor.width'] = 1
    mpl.rcParams['ytick.minor.visible'] = True
    mpl.rcParams['xtick.major.pad'] = 8
    mpl.rcParams['ytick.major.pad'] = 6
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams["contour.negative_linestyle"] = 'dotted'
    # mpl.rcParams['font.family'] = 'sans-serif'
    # mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.use('tkagg')


mplchange()
# mpl.rcParams.update({'font.size': 23})
# mpl.rcParams['xtick.major.size'] = 20
# mpl.rcParams['ytick.major.size'] = 20
# mpl.rcParams['xtick.major.width'] = 1.5
# mpl.rcParams['ytick.major.width'] = 1.5
# mpl.rcParams['xtick.minor.size'] = 12
# mpl.rcParams['ytick.minor.size'] = 12
# mpl.rcParams['xtick.minor.width'] = 1
# mpl.rcParams['ytick.minor.width'] = 1
# mpl.rcParams['xtick.major.pad'] = 8
# mpl.rcParams['ytick.major.pad'] = 6

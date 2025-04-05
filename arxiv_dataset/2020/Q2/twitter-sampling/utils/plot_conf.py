from collections.abc import Iterable
from .vars import ColorPalette


def concise_fmt(x, pos):
    if abs(x) // 10000000000 > 0:
        return '{0:.0f}B'.format(x / 1000000000)
    elif abs(x) // 1000000000 > 0:
        return '{0:.1f}B'.format(x / 1000000000)
    elif abs(x) // 10000000 > 0:
        return '{0:.0f}M'.format(x / 1000000)
    elif abs(x) // 1000000 > 0:
        return '{0:.1f}M'.format(x / 1000000)
    elif abs(x) // 10000 > 0:
        return '{0:.0f}K'.format(x / 1000)
    elif abs(x) // 1000 > 0:
        return '{0:.1f}K'.format(x / 1000)
    else:
        return '{0:.0f}'.format(x)


def hide_spines(axes):
    if isinstance(axes, Iterable):
        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    else:
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)

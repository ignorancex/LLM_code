
# external imports
from dataclasses import dataclass
from matplotlib.cm import get_cmap

'''
PlotAttribute: A class to store plot attributes.

Created on: Mar 28 2024

@author: Jonas Hall
'''
class PlotAttribute:
    def __init__(
            self, 
            c='black', 
            ls='-', 
            lw=2, 
            m=None, 
            ms=None, 
            a=1, 
            cm=None,
            aa=None
            ) -> None:
        self.color = c
        self.linestyle = ls
        self.linewidth = lw
        self.marker = m
        self.markersize = ms
        self.alpha = a
        self.cmap = cm
        self.antialiased = aa

    def getAttributes(self) -> dict:
        return {
            'color': self.color,
            'linestyle': self.linestyle,
            'linewidth': self.linewidth,
            'marker': self.marker,
            'markersize': self.markersize,
            'alpha': self.alpha,
            'cmap': self.cmap,
            'antialiased': self.antialiased,
        }


'''
PlotAttributes: Fixed plot attributes for different PM objects

Created on: Mar 28 2024

@author: Jonas Hall
'''
@dataclass
class PlotAttributes:
    agent = PlotAttribute(c='navy')
    target = PlotAttribute(c='red', m='o', ms=10)
    vector_field = PlotAttribute(c='black', a=0.3)
    cmap = 'viridis'
    sensor_quality = PlotAttribute(aa=True,a=0.4,cm=cmap,c=None,ls=None,lw=None)
    partition = PlotAttribute(c='black')
    partition_background = PlotAttribute(
        c=get_cmap(sensor_quality.cmap)(0.001), 
        a=sensor_quality.alpha
    )
    obstacle_background = PlotAttribute(
        c='black'
    )
    target_colors = [
        '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', 
        '#984ea3', '#999999', '#e41a1c', '#dede00'
    ]
    
    phi = PlotAttribute(c='forestgreen', m='X', ms=7.5, ls='')
    psi = PlotAttribute(c='yellow', m='p', ms=10, ls='')

    edge = PlotAttribute(c='black', ls='-', lw=2, a=0.5)

    u1_switch = PlotAttribute(c='salmon', ls='-')
    u2_switch = PlotAttribute(c='mediumaquamarine', ls='--')
    u_norm_switch = PlotAttribute('black', ls='-', lw=3)
    u1_monitor = PlotAttribute(c=u1_switch.color, ls=u1_switch.linestyle)
    u2_monitor = PlotAttribute(c=u2_switch.color, ls=u2_switch.linestyle)
    u_norm_monitor = PlotAttribute(
        c=u_norm_switch.color, 
        ls=u_norm_switch.linestyle, 
        a=u_norm_switch.alpha, 
        lw=u_norm_switch.linewidth
    )

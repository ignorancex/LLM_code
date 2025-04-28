
# external imports
import matplotlib.pyplot as plt
from pdfCropMargins import crop
import os


'''
Exporter: A class for export automation of matplotlib figures.

Created on: Mar 28 2024

@author: Jonas Hall       
'''
class Exporter:
    def __init__(self) -> None:
        self.WIDTH = 7
        self.HEIGHT = 7
        self.DPI = 300
        self.FONTSIZE = 14
        self.EXT = '.pdf'
        self.DIR = ''
        self.CROP = True
        self.TRANSPARENT = False

    def prepareDir(self, out_dir : str) -> None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def export(self, name, fig = None) -> None:
        self.prepareDir(self.DIR)
        file = os.path.join(self.DIR, name + self.EXT)
        
        if fig is not None:
            fig.set_size_inches(self.WIDTH, self.HEIGHT)

        plt.rcParams['font.size'] = self.FONTSIZE

        plt.savefig(file, bbox_inches='tight', dpi=self.DPI, transparent=self.TRANSPARENT)
        
        if self.CROP:
            self.cropMargins(file)

    def cropMargins(self, file) -> None:
        if self.EXT == '.pdf':
            crop(["-p", "20", "-u", "-s", str(file), "-o", str(self.DIR)])


'''
extendKwargs: Extend a dictionary with keyword arguments.

Created on: Mar 28 2024

@author: Jonas Hall

@details: Returns a new dictionary containing all keyword arguments in kwargs
          together with all non-None key-value pairs in p that don't exist in 
          kwargs.
'''
def extendKeywordArgs(p : dict, **kwargs) -> dict:
    eka = kwargs.copy()
    for key in p.keys():
        if not key in eka.keys() and p[key] is not None:
            eka[key] = p[key]
    return eka

'''
PlotObject: A container for matplotblib objects.

Created on: Mar 28 2024

@author: Jonas Hall

@details: This container class can be used to store matplotlib objects to 
    simplify their removal later on.
'''
class PlotObject:
    def __init__(self, *args) -> None:
        self._objs = []
        self.add(*args)

    def add(self, *args) -> None:
        for obj in args:
            if isinstance(obj, PlotObject):
                self.add(obj._objs)
            else:
                try:
                    for o in obj:
                        self.add(o)
                except:
                    self._objs.append(obj)

    def remove(self) -> None:
        for obj in self._objs:
            obj.remove()
        self._objs.clear()


'''
getAxes: Get the current axis.

Created on: Apr 1 2024

@author: Jonas Hall

@details: Returns the current axis if ax is None, otherwise returns ax.
'''
def getAxes(ax : plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    if not isinstance(ax, plt.Axes):
        raise ValueError("Expected ax to be of type matplotlib.pyplot.Axes.")
    return ax
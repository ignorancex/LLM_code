
from matplotlib import cm
from matplotlib.colors import ListedColormap

from .Filter import Filter


class ColorMap(Filter):
    """ColorMap

    Transforms the image from classes to rgb using a Matplotlib colormap.
    
    Example:
    - Type: ColorMap
      cmap: tab20
    """
    def __init__(self, cmap, add_zero=False, **args):
        super().__init__(**args)
        self._cmap = cm.get_cmap(cmap or 'tab20')
        if add_zero:
            self._cmap = ListedColormap(((0,0,0),*self._cmap.colors))

    def apply_one(self,  image, i, prepared):
        return self._cmap(image)
        
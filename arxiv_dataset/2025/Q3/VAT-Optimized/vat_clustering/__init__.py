from .vat import vat, plot_vat
from .optimization import vat_optimized

# Import Cython VAT only if available (to avoid import errors if Cython is not installed)
try:
    from .vat_cython import vat_cython
    __all__ = ["vat", "plot_vat", "vat_optimized", "vat_cython"]
except ImportError:
    __all__ = ["vat", "plot_vat", "vat_optimized"]

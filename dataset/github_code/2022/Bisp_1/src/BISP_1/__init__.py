#from .BISP_1 import *

from .Bisp import *
from .Stars import *
from .Decomp import *
from .Priors import *

from pathlib import Path


path_to_here = Path(__file__).parent.absolute()
path_to_tomlfile = path_to_here / ".." / ".." / "pyproject.toml"

version = "unknown"

with open(path_to_tomlfile) as fp:
    for line in fp.read().splitlines():
        if line.startswith('version'):
            delim = '"' if '"' in line else "'"
            version = str(line.split(delim)[1])

__version__ = version

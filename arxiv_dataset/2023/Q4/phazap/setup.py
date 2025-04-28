import setuptools
from pathlib import Path

import re
VERSIONFILE="phazap/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


setuptools.setup(
    name="phazap",
    version=verstr,
    author="Jose María Ezquiaga",
    author_email="jose.ezquiaga@nbi.ku.dk",
    description="Gravitational wave phase reconstruction for low-latency identification of strongly lensed signals",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=[
        "phazap",
    ],
    install_requires=[
        "bilby",
        "bilby_pipe",
        "pesummary",
        "getdist",
        "tqdm",
        "configargparse",
        "numpy >= 1.21.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    python_requires='>=3.7',
)

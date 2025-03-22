import os
import re
from setuptools import setup, find_packages


def read(filename):
    f = open(filename)
    r = f.read()
    f.close()
    return r

ver = re.compile("__version__ = \"(.*?)\"")
#m = read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "refann", "__init__.py"))
#m = read(os.path.join(os.path.dirname(__file__), "refann", "__init__.py"))
m = read(os.path.join(os.getcwd(), "cmbnncs", "__init__.py"))
version = ver.findall(m)[0]



setup(
    name = "cmbnncs",
    version = version,
    keywords = ("pip", "CNN"),
    description = "CMB Neural Network Component Separator",
    long_description = "",
    license = "MIT",

    url = "",
    author = "Guojian Wang",
    author_email = "gjwang2018@gmail.com",

#    packages = find_packages(),
    packages = ["cmbnncs", "examples"],
    include_package_data = True,
    data_files = [],
    platforms = "any",
    install_requires = []
)


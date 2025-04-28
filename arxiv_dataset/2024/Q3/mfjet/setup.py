from os import path
from setuptools import setup

# get version
version_file = path.join(path.abspath(path.dirname(__file__)), 'mfjet', 'version.py')
with open(version_file, 'r') as file:
    exec(file.read())

# fill out dynamic entries
setup(
    version = __version__
)

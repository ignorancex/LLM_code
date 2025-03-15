"""ProtoDUNE raw data denoising with DL"""
# Put here functions or variables to be exposed
# that way the log system is imported from the very beginning
from importlib.metadata import metadata
from dunedn.configdn import PACKAGE

__version__ = metadata(PACKAGE)["version"]

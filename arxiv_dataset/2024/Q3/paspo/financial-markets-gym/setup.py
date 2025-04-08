from setuptools import setup

setup(name="financial_markets_gym", #be aware of _ and -, the root folder is - while the subfolder is _
      version="1.0.0",
      install_requires=["gym", "numpy", "scikit-learn", "hmmlearn"]
)
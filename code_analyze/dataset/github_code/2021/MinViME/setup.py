# -*- coding: utf-8 -*-
 
"""setup.py: setuptools control."""
 
import re
from setuptools import setup
 
version = re.search(
        '^__version__\s*=\s*"(.*)"',
        open('minvime/__init__.py').read(),
        re.M
    ).group(1)
 
with open("README.md", "rb") as f:
    long_descr = f.read().decode("utf-8")

setup(
    name = "minvime",
    packages = ["minvime"],
    license = "MIT",
    install_requires = ['pandas>=0.25.3', 'numpy>=1.16.4', 'Flask>=2.2.2', 'matplotlib'],
    entry_points = {
        "console_scripts": ['minvime = minvime.minvime:main']
    },
    include_package_data=True,
    version = version,
    description = "Python Application for Estimating Minimum Viable Model Performance.",
    long_description = long_descr,
    long_description_content_type='text/markdown',
    author = "John Hawkins",
    author_email = "john@getting-data-science-done.com",
    url = "http://getting-data-science-done.com",
    project_urls = {
        'Documentation': "https://minvime.readthedocs.io",
        'Source': "https://github.com/john-hawkins/minvime",
        'Tracker': "https://github.com/john-hawkins/minvime/issues" 
      }
    )


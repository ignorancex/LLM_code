##### Measuring Diagnostic Test Performance Using an Imperfect Reference Test: A Partial Identification Approach #####
### Package installation script
## Developed by: Filip Obradović. Email: obradovicfilip@u.northwestern.edu

""" This script installs all necessary packages."""

import sys
import subprocess
import pkg_resources

required = {'numpy', 'scipy', 'statsmodels', 'matplotlib', 'joblib'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
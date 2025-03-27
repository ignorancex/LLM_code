# -*- coding: utf-8 -*-
  
"""INFO

Set the package information for `CorrMatrix_ABC'

:Author: Martin Kilbinger, <martin.kilbinger@cea.fr>

:Date: 25/01/2021

:Package: CorrMatrix_ABC
"""

# Release version
version_info = (1, 2, 0)
__version__ = '.'.join(str(c) for c in version_info)

# Package details
__author__ = 'Martin Kilbinger, Emille Ishida, Jessi Kehe'
__email__ = 'martin.kilbinger@cea.fr'
__year__ = 2021
__url__ = 'https://github.com/emilleishida/CorrMatrix_ABC'
__description__ = 'Correlation matrix and ABC'

# Dependencies
__requires__ = ['numpy', 'scipy', 'astropy', 'statsmodels', 'pystan']

# Default package properties
__license__ = 'GNU General Public License (GPL>=3)'
__about__ = ('{} \n\n Author: {} \n Email: {} \n Year: {} \n {} \n\n'
             ''.format(__name__, __author__, __email__, __year__,
                       __description__))
__setup_requires__ = ['pytest-runner', ]
__tests_require__ = ['pytest', 'pytest-cov', 'pytest-pep8']

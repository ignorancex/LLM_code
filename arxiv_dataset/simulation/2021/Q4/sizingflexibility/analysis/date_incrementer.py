"""
__author__ = "Babu Kumaran Nalini"
__copyright__ = "2021 TUM-EWK"
__credits__ = []
__license__ = "GPL v3.0"
__version__ = "1.0"
__maintainer__ = "Babu Kumaran Nalini"
__email__ = "babu.kumaran-nalini@tum.de"
__status__ = "Development"
"""

from datetime import datetime, timedelta

def date_incrementer(i):
    base = datetime.strptime('2019-01-01', "%Y-%m-%d")
    c_date = base + timedelta(days=i)
    date = c_date.strftime('%Y-%m-%d')
    return date
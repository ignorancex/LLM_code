import os
import numpy as np
from astropy.table import Table

from paths import (
    data as data_dir,
)

__all__ = ['get_age_ranges']


def get_age_ranges(age_ranges):
    """
    Extracts the average age per age bin and associated error.

    Returns
    -------
    ages: np.array
       Array of `ages[:,0] = average age`; `ages[:,1] = lower age error`;
       `ages[:,2] = upper age error`.
    """
    tab = Table.read(os.path.join(data_dir,
                                  'final_flare_catalog_v6.csv'),
                     format='csv')

    ages = np.zeros((len(age_ranges), 3))

    for i in range(len(age_ranges)):
        temp = tab[(tab['age']>=age_ranges[i][0]) &
                   (tab['age']<age_ranges[i][1])]
        avg = np.nanmean(temp['age'])
        maxa = temp[temp['age']==np.nanmax(temp['age'])][0]['eage']
        mina = temp[temp['age']==np.nanmin(temp['age'])][0]['eage']
        ages[i][0] = avg
        ages[i][1] = (avg - np.nanmin(temp['age']))
        ages[i][2] = (np.nanmax(temp['age'])-avg)

    return ages

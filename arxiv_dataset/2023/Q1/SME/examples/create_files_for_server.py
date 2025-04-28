# -*- coding: utf-8 -*-
from pysme.large_file_storage import setup_atmo, setup_nlte

lfs_atmo = setup_atmo()
lfs_nlte = setup_nlte()

for lfs in [lfs_atmo, lfs_nlte]:
    pointers = lfs.generate_pointers()
    pass

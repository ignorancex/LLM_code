# -*- coding: utf-8 -*-
from pysme.large_file_storage import setup_atmo, setup_nlte

lfs_atmo = setup_atmo()
lfs_nlte = setup_nlte()

for name, lfs in zip(["atmospheres", "nlte"], [lfs_atmo, lfs_nlte]):
    for key in lfs.pointers.keys():
        print(key)
        lfs.get(key)
    pointers = lfs.generate_pointers()
    lfs.create_pointer_file(f"datafiles_{name}.json")
    lfs.move_to_cache()
    pass

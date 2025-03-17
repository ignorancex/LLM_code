#!/usr/bin/env ipython

import json
import pandas as pd
from glob import glob

if __name__ == '__main__':
    print("instance_id, N, E, R, wW, wH")
    for infile in glob("./instances/orig/UDMIS*.orig.json"):
        with open(infile, 'r') as infilehandle:
            js = json.load(infilehandle)

        print(f"{js['description']['instance_id']}, {len(js['nodes'])}, {len(js['edges'])}, {js['description']['R']}, {js['description']['wheight']},{js['description']['wwidth']}")

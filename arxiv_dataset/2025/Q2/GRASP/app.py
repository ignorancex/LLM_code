import argparse
from graph_construction.parser import create_parser
from graph_construction import *

if __name__ == "__main__":
    config = create_parser()
    config = config.parse_args()
    print('Magnifications requested to working on:', config.mags)
    gc = graph_construction(config)
    gc.run()

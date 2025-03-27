import argparse
import yaml
import pandas as pd

from openforest.scripts.openforest_database import OpenForestDatabase
from openforest.utils import OPENFOREST_HOME
from openforest.utils.functions import get_dataset_name

def main():
    args = get_dataset_name()
    dataset_name = args['dataset_name']
    database = OpenForestDatabase()
    database.delete_row(dataset_name)
    database.erase

if __name__ == '__main__':
    main()

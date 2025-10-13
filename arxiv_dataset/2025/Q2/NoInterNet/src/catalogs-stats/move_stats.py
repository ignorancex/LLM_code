import os
import sys
import pandas as pd
import argparse

def main_fiducial(ns):
    
    paths = pd.read_csv(ns.input_csv)

    for i, row in paths.iterrows():

        path = os.path.normpath(row['Pk_int'])
        Ndir = path.split(os.sep)[-2]

        new_dir = os.path.join(ns.output, Ndir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        Pk_int = row['Pk_int']
        Bk_int = row['Bk_int']
        os.system('cp ' + Pk_int + ' ' + os.path.join(new_dir, 'Pk_pylians-dz.dat'))
        os.system('cp ' + Bk_int + ' ' + os.path.join(new_dir, 'Bk_6k_pyspectrum-dz.dat'))
            
        Pk_true = row['Pk_true']
        os.system('cp ' + Pk_true + ' ' + os.path.join(new_dir, 'Pk_pylians-no-dz.dat'))
        
def main_cosmo(ns):
    
    paths = pd.read_csv(ns.input_csv)

    if ns.who == 'LH':
        N = 2000
    elif ns.who == 'BSQ':
        N = 2**15

    for i, row in paths.iterrows():

        path = os.path.normpath(row['Pk_int'])
        Ndir = path.split(os.sep)[-2]
        ndir = int(i/N)

        new_dir = os.path.join(ns.output, Ndir, str(ndir))
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        Pk_int = row['Pk_int']
        Bk_int = row['Bk_int']
        os.system('cp ' + Pk_int + ' ' + os.path.join(new_dir, 'Pk_pylians-dz.dat'))
        os.system('cp ' + Bk_int + ' ' + os.path.join(new_dir, 'Bk_6k_pyspectrum-dz.dat'))
        
        Pk_true = row['Pk_true']
        if ns.type == 'inbox':
            os.system('cp ' + Pk_true + ' ' + os.path.join(new_dir, 'Pk_pylians-no-dz.dat'))
        else:
            os.system('cp ' + Pk_true + ' ' + os.path.join(ns.output, Ndir, 'Pk_pylians-no-dz.dat'))
        

if __name__ == '__main__':
    desc = 'Build Pk dir structure consistent with catalogs'

    parser = argparse.ArgumentParser(description=desc)

    # required arguments
    group = parser.add_argument_group('required arguments')

    h = 'path to csv file'
    group.add_argument('--input-csv', type=str, help=h, required=True)

    h = 'output path to copy the files'
    group.add_argument('--output', type=str, help=h, required=True)

    h = 'what set of simulation Pk to move'
    group.add_argument('--who', type=str, help=h, choices=['fiducial', 'LH', 'BSQ'], required=True)

    h = 'type of interlopers'
    group.add_argument('--type', type=str, help=h, choices=['inbox', 'outbox'], required=True)

    # and go!
    if parser.parse_args().who == 'fiducial':
        main_fiducial(parser.parse_args())
    else:
        main_cosmo(parser.parse_args())
###############################################################################
# get_science_exposures.py
###############################################################################
#
# Extract the correct list of science exposures from the pps html files
#
###############################################################################


from bs4 import BeautifulSoup
import pandas as pd
import glob
import sys

# Parse keyword arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--xmmdata',action='store',dest='xmmdata',
                    default='False',type=str)
parser.add_argument('--obsID',action='store',dest='obsID',
                    default='False',type=str)
results = parser.parse_args()
xmmdata = results.xmmdata
obsID = results.obsID

obs_dir = xmmdata + '/' + obsID

# Store the file path to the PPS summary file
# Open it in beautiful soup for processing
ppssum_loc = glob.glob(obs_dir + '/pps/*PPSSUM*.HTM')[0]
with open(ppssum_loc) as f:
    BS = BeautifulSoup(f, 'html.parser')

# Parse the HTML for the table listing the EPIC exposures
# This is the first widetable
exp_table = BS.find('div', {'id':'widetable'}).find('table')

# Store each table header as a Pandas DataFrame
try:
    exp_df = pd.read_html(str(exp_table),header=0,index_col=0)[0]
except ValueError:
    sys.exit('There were no exposures recorded by the EPIC camera.')

# Iterate through each row = science exposure
for row in exp_df.itertuples():
    # Use these to calculate the exposure name
    inst = str(row[0][1:])
    expid = str(row[1])

    # For PN, only certain modes are allowed with ESAS
    # Store the exposures with the correct mode in a file
    if inst == 'PN':
        mode = row[3]
        if (mode == 'PrimeFullWindowExten') or (mode == 'PrimeFullWindow'):
            pn_exp_file = obs_dir + '/pn_exposures.txt'
            with open(pn_exp_file,'a') as pef:
                pef.write(expid+'\n')
    # Store the MOS exposures in a file
    elif inst[:-1] == 'MOS':
        mos_exp_file = obs_dir + '/mos_exposures.txt'
        prefix = inst[-1]+expid
        with open(mos_exp_file,'a') as m1ef:
            m1ef.write(prefix+'\n')

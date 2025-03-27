#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 2024

A snippet of code for merging sequentially the data from a specified 
Excel sheet from Excel files into a single Excel file (the Sheet will have
the same name). If the 'w1' and 'w2' arguments are not specified, then
the program will read all files from the input directory -- note that all 
files need to have the Excel Sheet specified by the 'sheet' argument.
If the two input arguments 'w1' and 'w2' are provided, then only data from 
the period from the first week 'w1' to the last week 'w2' will be considered.

@author: vesseln1
"""

#!/usr/bin/env python
# coding: utf-8
# nvv


__title__			 = 'Mobile traffic forecasting'
__description__		 = 'Merges sequentually the data from files into a dataset.'
__version__			 = '2.0.0'
__date__			 = 'August 2024'
__author__			 = 'Natalia Vesselinova'
__author_email__	 = 'natalia.vesselinova@aalto.fi'
__institution__ 	 = 'Alto University'
__department__		 = 'Mathematics and Systems Analysis'
__url__				 = 'https://github.com/nvassileva/DataMatters/'
__license__          = 'CC BY 4.0'




import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', '-d', required=True, type=str, help="The path to the root directory with the data to merge.")
parser.add_argument('--file_name', '-f', required=False, type=str, help='The name of the file that will contain all data, from all files in dataroot, merged sequentially.', default='pems.csv')
parser.add_argument('-w1', required=False, type=int, help='The first week of data.')    
parser.add_argument('-w2', required=False, type=int, help='The last week of data.' )
parser.add_argument('--sheet', '-s',  required=False, type=str, help='The Excel Sheet where the data is stored (read from).', default='MLData' )
args = parser.parse_args()


data_root   = args.data_root
file_name   = args.file_name
sheet       = args.sheet
w1          = args.w1 
w2          = args.w2

# A function for extracting the required sheet and for concatenating it 
# to the passed pre-existing dataframe 
def extractSheet(file, sheet):
    """
    Extracts the data from the passed Excel sheet and returns it as a dataframe

    Input parameters
    file        : the file containg the Excel Sheet with the data
    sheet       : the name of the Excel Sheet

    Returns     : the Excel Sheet data saved in a dataframe
    """
    xl = pd.ExcelFile(file)
    dfs= {sh:xl.parse(sh) for sh in xl.sheet_names}
      
    return dfs[sheet]



def concatenateFiles(df, files):
    """
    Concatenates dataframes returned by extractSheet() into a single dataframe

    Input parameters
    df          : a pre-existing dataframe
    files       : the files in the data_root directory
        

    Returns     : the pre-existing dataframe containing the merged dataframes
    """
    if w1 is not None and w2 is not None:
        for num in range(w1, (w2 + 1)):
            for file in files:
                if file.endswith(str(num) + '.xlsx'):
                    print(file)
                    df = pd.concat([df, extractSheet(file, sheet)])
    else:
        for file in sorted(files, key=lambda x: x.split(".")[0]):
            if file.endswith('.xlsx'):
                print(file)
                df = pd.concat([df, extractSheet(file, sheet)])
    
    return df


# Change the directory to the data folder
os.chdir(data_root)
files = os.listdir()
print(files)

df_total = pd.DataFrame()
df_total = concatenateFiles(df_total, files)
df_total = df_total.assign(Index=range(len(df_total))).set_index('Index')
print(df_total.shape)

# df_total.to_excel(file_name, sheet_name=sheet, index=False)
# Order them so that the predicted variable (the target) 'Total Calls' is at the last postion
df_total = df_total.loc[:, ['Flow (Veh/5 Minutes)', 'Speed Level', 'Total Number Calls']]
df_total.to_csv(file_name, index=False)

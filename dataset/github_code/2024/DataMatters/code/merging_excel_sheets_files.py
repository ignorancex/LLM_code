#!/usr/bin/env python3
# coding: utf-8
# nvv

"""
Created on Tue Aug 21 2024

This code reads data from the relevant Excel files and generates a pems.csv file
with the data used by the machine learning algorithm. 
Merges sequentially data from a set of specified Excel sheets into 
a single file. If there is more than one Excel file in the indicated directory, 
then all files are read. If the input arguments 'w1' and 'w2' are given, then 
only data from the period from the first to the last week is merged. 
Files, in that case, must be named in terms of (consequitive) week numbers. 

@author: vesseln1
"""



__title__			 = 'Mobile traffic forecasting'
__description__		 = 'Merges sequentually the data from files into a dataset.'
__version__			 = '1.0.0'
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
parser.add_argument('--sheets', '-s',  required=False, type=str, help='The name of the Excel sheets where the data is stored', nargs='+', default=['MLData', 'Data', 'AdjustedHOsfromBS1'])
parser.add_argument('--variables_names', '-v',  required=False, type=str, help='The name of the Excel columns (variables) of interest', nargs='+', default=['Flow (Veh/5 Minutes)', 'Speed Level', 'Total New Calls', 'Total HO Calls', 'Total Number Calls'])
args = parser.parse_args()


data_root   = args.data_root
file_name   = args.file_name
sheets      = args.sheets
var_names   = args.variables_names
w1          = args.w1 
w2          = args.w2


def extractSheet(file, sheet):
    """
    Extracts the data from the passed Excel sheet and returns it as a dataframe

    Input parameters
    file        : the file containg the Excel sheet with the data
    sheet       : the name of the Excel sheet

    Returns     : the Excel sheet data saved in a dataframe
    """
    xl = pd.ExcelFile(file)
    dfs= {sh:xl.parse(sh) for sh in xl.sheet_names}
    if sheet == 'Data':
        return dfs[sheet]['Total New Calls']
    elif sheet == 'AdjustedHOsfromBS1':
        return dfs[sheet]['Total HO Calls']
    else:
        return dfs[sheet].drop(columns=['Index'])



def concatenateSheets(df, sheets, file):
    """
    Concatenates dataframes returned by extractSheet() into a single dataframe

    Input parameters
    df          : a pre-existing dataframe
    files       : the files in the data_root directory
        

    Returns     : the pre-existing dataframe containing the merged dataframes
    """
    for sheet in sheets:
        df = pd.concat([df, extractSheet(file, sheet)], axis=1)
        
    return df



def concatenateFiles(df_total, df, sheets, files):
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
                    df_total = pd.concat([df_total, concatenateSheets(df, sheets, file)])
    else:
        for file in sorted(files, key=lambda x: x.split(".")[0]):
            if file.endswith('.xlsx'):
                print(file)
                df_total = pd.concat([df_total, concatenateSheets(df, sheets, file)])
    
    return df_total

# Change the directory to the data folder
os.chdir(data_root)
files = os.listdir()

df_total = pd.DataFrame()
df = pd.DataFrame()

df_total = concatenateFiles(df_total, df, sheets, files)
df_total = df_total.loc[:, var_names]
if w1 is not None and w2 is not None:
    n1 = str(w1)
    n2 = str(w2)
    df_total.to_csv(file_name + '_' + n1 + '-' + n2 + '.' + 'csv', index=False)
else:
    df_total.to_csv(file_name, index=False)



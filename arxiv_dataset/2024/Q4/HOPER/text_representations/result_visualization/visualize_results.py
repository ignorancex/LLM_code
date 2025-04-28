#This is the main function that parses command-line arguments and runs the analysis.
import argparse
import create_figures as cf
import create_significance as cs
import pandas as pd
import tqdm

parser = argparse.ArgumentParser(description='A ')
parser.add_argument("-f","--figures", action='store_true', help="Create figures from results")
parser.add_argument("-s", "--significance", action='store_true',  help="Create significance tables from results")
parser.add_argument("-rfp", "--resultfilespath", required=True,  help="Path for the result files")
parser.add_argument("-a", "--all", action='store_true',  help="Create both figures and significance tables")

try:
    args = parser.parse_args()
    if not (args.figures or args.significance):
            parser.error('At least one option should be selected!')
except:
    parser.print_help()

print(args)

if args.figures or args.significance or args.all:
    print("Loading results... \n\n")
    
 
if args.figures or args.all:
    print("\n\n Creating figures...\n")
    cf.result_path = args.resultfilespath
    cf.create_figures()
if args.significance or args.all:
    print("\n\nCreating significance tables...\n")
    cs.result_path = args.resultfilespath
    cs.create_significance_tables()



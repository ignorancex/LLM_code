import pandas as pd
from pathlib import Path
import sys
import os

"""

To run this file, type in the terminal

python3 Match_rs_id.py "/home/user/Exposure.csv" "/home/user/annotation.csv" 

Here,

Compulsory argument

1. 1st argument "/home/user/Exposure.csv", path to the exposure data file as downloaded from GTEx

2. 2nd argument "/home/user/annotation.csv", path to the SNPs annotation file as downloaded from GTEx


"""


def check_valid(user_input, check):
    out = None
    allowed_extensions = [".txt", ".csv"]
    if check == "path":
        if not (Path(user_input).exists() and Path(user_input).is_file and Path(
                user_input).suffix in allowed_extensions):
            print(
                "Provide a valid path with .txt or .csv extension")
            sys.exit(1)
        else:
            out = user_input
    else:
        print("No parameter given")
    return out


try:
    file_exp = sys.argv[1]
    file_exp = check_valid(file_exp, "path")
    file_ann = sys.argv[2]
    file_ann = check_valid(file_ann, "path")

except IndexError:
    print("You failed to either provide the minimum or maximum number of parameters required to run this code. Please "
          " read the documentation for more details")
    sys.exit(1)


def match(expression_file, annotation_file):
    path_original = os.path.dirname(expression_file)
    name_exp = os.path.splitext(os.path.basename(expression_file))[0]
    gene_expression_data = pd.read_csv(expression_file, sep='\t', low_memory=False)
    annotation_data = pd.read_csv(annotation_file, sep='\t', low_memory=False)
    # Merge on the common column
    merged_data = pd.merge(gene_expression_data, annotation_data, on='variant_id', how='inner')

    # Select the relevant columns (GTEx variant ID and rs ID)
    mapped_data = merged_data[
        ['phenotype_id', 'variant_id', 'rs_id_dbSNP151_GRCh38p7', 'maf', 'ref', 'alt', 'slope', 'slope_se',
         'pval_beta', 'chr', 'variant_pos']]

    # Define a dictionary for column renaming
    column_mapping = {
        'phenotype_id': 'exposure',
        'variant_id': 'variant_id',
        'rs_id_dbSNP151_GRCh38p7': 'SNP',
        'maf': 'eaf.exposure',
        'ref': 'other_allele.exposure',
        'alt': 'effect_allele.exposure',
        'slope': 'beta.exposure',
        'slope_se': 'se.exposure',
        'pval_beta': 'pval.exposure',
        'chr': 'chr',
        'variant_pos': 'pos'
    }
    # Rename the columns
    mapped_data = mapped_data.rename(columns=column_mapping)
    # Copy 'phenotype_id' to a new column 'id_exposure'
    mapped_data['id.exposure'] = mapped_data['exposure']
    mapped_data.to_csv(path_original + "/" + name_exp + "_annotated.csv", sep=",", index=False)
    print("Annotated Data has been saved in the directory " + path_original)


match(file_exp, file_ann)

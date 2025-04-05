"""
Creates the ARF/MOSF, COPD/CHF/Cirrhosis, Cancer, and Coma subsets of the
SUPPORT dataset using the 14 features used in the DeepSurv paper; in our
paper we refer to these 4 subsets as SUPPORT1, SUPPORT2, SUPPORT3, and
SUPPORT4 respectively

The original SUPPORT dataset can be obtained from https://hbiostat.org/data/
courtesy of the Vanderbilt University Department of Biostatistics (we
specifically downloaded the support2csv.zip file; note that what they call
SUPPORT2 should not be confused with what we call SUPPORT2: they refer to the
larger dataset they have collected as SUPPORT2, which is what we split into
four subsets; in our paper, we refer to the COPD/CHF/Cirrhosis subset as
SUPPORT2 for the sake of brevity)

Missing values are imputed for these 14 features specifically only for 2 of
them that have suggested imputation values in the official SUPPORT
documentation ('wblc' and 'crea'; the official documentation does mention
imputation values for some other features that are not among the 14 we are
using)

Author: George H. Chen
"""
import csv
import os
import numpy as np
import pandas as pd


# same 14 features as used in DeepSurv paper along with observed outcomes
# 'd.time' and 'death'
columns_to_keep = \
    ['age', 'sex', 'race', 'num.co', 'diabetes', 'dementia', 'ca', 'meanbp',
     'hrt', 'resp', 'temp', 'wblc', 'sod', 'crea', 'd.time', 'death']


def parse_feature(x, feature_name):
    if feature_name in {'age', 'meanbp', 'hrt', 'resp', 'temp', 'wblc', 'sod',
                        'crea', 'd.time'}:
        if x != '':
            if float(x) == int(float(x)):
                return int(float(x))
            else:
                return float(x)
        else:
            # the official SUPPORT documentation has suggested "normal"
            # imputation values for 'wblc' and 'crea'
            if feature_name == 'wblc':
                return 9
            elif feature_name == 'crea':
                return 1.01
            else:
                return np.nan
    elif feature_name in {'sex', 'race', 'ca'}:
        return x
    elif feature_name in {'num.co', 'diabetes', 'dementia'}:
        if x != '':
            return int(x)
        else:
            return np.nan
    elif feature_name == 'death':
        return int(x)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    features = None
    dzclasses = set()
    feature_to_idx = {}
    csv_rows = []
    with open('support2.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = True
        for row in csv_reader:
            if header:
                header = False
                features = row
                idx = 0
                for feature in features:
                    feature_to_idx[feature] = idx
                    idx += 1
                continue
            new_row = [x.strip() for x in row[1:]]
            csv_rows.append(new_row)
            dzclasses.add(new_row[feature_to_idx['dzclass']])
    dzclasses = sorted(list(dzclasses))

    output_csv_writers = {}
    for dzclass in dzclasses:
        f = open('support_%s.csv' % (dzclass.replace('/', '_')), 'w')
        output_csv_writers[dzclass] = csv.writer(f, delimiter=',',
                                                 quoting=csv.QUOTE_NONNUMERIC)
        output_csv_writers[dzclass].writerow(
            [features[feature_to_idx[col]]
             for col in columns_to_keep])

    for row in csv_rows:
        new_row = [parse_feature(row[feature_to_idx[col]], col)
                   for col in columns_to_keep]
        dzclass = row[feature_to_idx['dzclass']]
        output_csv_writers[dzclass].writerow(new_row)

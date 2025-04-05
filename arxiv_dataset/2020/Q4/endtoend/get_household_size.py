import pandas as pd
import numpy as np

# DATASET: Dataset: ESS8-2016, ed.2.1
full_df = pd.read_stata("data/ESS8GB.dta", index_col="idno", convert_categoricals=False)
# idno is now in pandas.index

# keep relevant variables (descriptions below)
household_df = full_df[['agea','yrbrn2','yrbrn3','yrbrn4','yrbrn5','yrbrn6','yrbrn7','yrbrn8','yrbrn9','yrbrn10','yrbrn11','yrbrn12','pspwght']]

household_df = household_df[household_df['agea'].notnull()]
# variable for number of people in household over 16
household_df['size_above_16'] = (household_df['agea']>=16)*1
for yr in range(2,13):
	household_df['size_above_16'] = household_df['size_above_16'] + (2016-household_df['yrbrn'+str(yr)]>=16 & household_df['yrbrn'+str(yr)].notnull())*1

# variable for ratio of weight to household size
household_df['weight_to_size_ratio'] = household_df['pspwght']/household_df['size_above_16']

# reweighting for household size
avg_household_size = household_df['pspwght'].sum()/household_df['weight_to_size_ratio'].sum() 

print(f"{avg_household_size:.2f}")

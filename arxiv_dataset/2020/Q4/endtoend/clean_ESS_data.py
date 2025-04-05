import pandas as pd
import numpy as np
import math

# suppress warnings
pd.options.mode.chained_assignment = None

# DATASET: Dataset: ESS8-2016, ed.2.1
full_df = pd.read_stata("data/ESS8GB.dta", index_col="idno", convert_categoricals=False)
# idno is now in pandas.index

# keep relevant variables (descriptions below)
clean_df = full_df[['gndr','agea','region','edulvlb','wrclmch','blgetmg','domicil','hhmmb','pspwght']]#,'rshipa2',
#'rshipa3','rshipa4','yrbrn2','yrbrn3','yrbrn4']]

# drop all people under 16
clean_df.drop(clean_df[clean_df['agea']<16].index, axis=0, inplace=True) # doing the opposite (keeping >= 16) automatically drops missings, which we don't want

# missing values report
print("\nMISSING VALUES REPORT:")
print("total number of respondents: "+str(len(clean_df)))
print(clean_df.isnull().sum())

# construct variables

####### gender #######
# Foundation: "gender" = Male / Female / Other
# ESS: "gndr" = Male (1) / Female (2) / No Answer (9)

print("\nconstructing GENDER...")

print("number people who were missing or said No answer: "+str(clean_df.isnull().sum()['gndr']+len(clean_df[clean_df['gndr']==9])))

def genders(series):
    if series==1:
        return "Male"
    elif series==2:
        return "Female"
    else:
        print("unassignable response found - assigning to missing")
        return float("NAN")

clean_df['gender'] = clean_df['gndr'].apply(genders)

####### age #######
# Foundation: "age_bucket" = 16-29 / 30-44 / 45-59 / 60+
# ESS: "agea" = (continuous), or 999 = not available


print("\nconstructing AGE...")

print("number people who were missing: "+str(clean_df.isnull().sum()['agea']))
print("number people who said no answer: "+str(len(clean_df[clean_df['agea']>=900])))

# Impute missings with spouse age data - HAS NO EFFECT - COMMENTED OUT
"""
# (I check only relations 2,3,4 because no one reported "spouse" for 5-12)
clean_df['spouse_age'] = (clean_df['rshipa2']==1)*(2016-clean_df['yrbrn2']) + (clean_df['rshipa3']==1)*(2016-clean_df['yrbrn3']) + (clean_df['rshipa4']==1)*(2016-clean_df['yrbrn4'])
print(clean_df['spouse_age'].value_counts())


# replace 0s with missings
def spouse_ages(series):
    if series==0:
        return float("NAN")
    else:
    	return series
#clean_df['spouse_age'] = clean_df['spouse_age'].apply(spouse_ages)

# if agea missing and spouse age is not, replace agea with spouse age
clean_df['agea'] = clean_df['agea'].fillna(clean_df['spouse_age'])

print("number people who are missing after impute: "+str(clean_df.isnull().sum()['agea']))
"""

def age_buckets(series):
    if series >= 900 or math.isnan(series):
        print("unassignable response found - assigning to missing")
        return float("NAN")
    elif 16 <= series <= 29:
        return "16-29"
    elif 30 <= series <= 44:
        return "30-44"
    elif 45 <= series <= 59:
        return "45-59"
    elif 60 <= series <= 900:
        return "60+"

clean_df['age_bucket'] = clean_df['agea'].apply(age_buckets)




####### location #######
# Foundation: "geo_bucket" = North East / North West/ Yorkshire and The Humber / East Midlands / West Midlands /
#                            East of England / London / South East / South West / Wales / Scotland / Northern Ireland
# ESS: "region"  = North East (UKC) / North West (UKD) / Yorkshire and the Humber (UKE) / East Midlands (England) (UKF) /
#                  West Midlands (England) (UKG) / East of England (UKH) / London (UKI) / South East (England) (UKJ) /
#                  South West (England) (UKK) / Wales (UKL) / Scotland (UKM) / Northern Ireland (UKN)
print("\nconstructing LOCATION...")

num_uninformative = clean_df.isnull().sum()['region']
print("Number of people missing: "+str(num_uninformative))

def geo_buckets(series):
    if series=="UKC":
        return "North East"
    elif series=="UKD":
        return "North West"
    elif series=="UKE":
        return "Yorkshire and The Humber"
    elif series=="UKF":
        return "East Midlands"
    elif series=="UKG":
        return "West Midlands"
    elif series=="UKH":
        return "East of England"
    elif series=="UKI":
        return "London"
    elif series=="UKJ":
        return "South East"
    elif series=="UKK":
        return "South West"
    elif series=="UKL":
        return "Wales"
    elif series=="UKM":
        return "Scotland"
    elif series=="UKN":
        return "Northern Ireland"
    else:
        print("unassignable response found - assigning to missing")
        return float("NAN")

clean_df['geo_bucket'] = clean_df['region'].apply(geo_buckets)



####### education #######
# Foundation: "edu_bucket" = No qualifications or Level 1 / Level 2 or Level 3 or apprenticeship or other / Level 4 and above
# ESS: "edulvlb" =  Other (5555) / Refusal (7777) / Don't know (8888) / No answer (9999)
#                   No qualifications or Level 1 -> {0,113}
#                    Level 2 or Level 3 or apprenticeship -> {129,212,213,221,222,223,229,311,312,313,321,322,323}
#                   Level 4 and above -> {412,413,421,422,423,510,520,610,620,710,720,800}

#Other, Refusal, Don't know, and No answer are encoded as other. Count these people:
print("\nconstructing EDUCATION...")
num_uninformative = clean_df.isnull().sum()['edulvlb']+len(clean_df[clean_df['edulvlb']>5000])
print("Number of people missing or said Other / Refusal / Don't know / No answer: "+str(num_uninformative))

def edu_buckets(series):
    if series in [0,113]:
        return "No Qualifications/Level 1"
    elif math.isnan(series) or series>5000 or 129 <= series <= 323:
        return "Level 2/Level 3/Apprenticeship/Other"
    elif 412 <= series <= 800:
        return "Level 4 and above"

clean_df['edu_bucket'] = clean_df['edulvlb'].apply(edu_buckets)


####### climate concern level #######
# Foundation: "climate_concern_level" = Very concerned / Fairly concerned / Not very concerned / not at all concerned / Other
# ESS: "wrclmch" = Not at all worried (1) / Not very worried (2) / Somewhat worried (3) / Very worried (4) /
#                  Extremely worried (5) / Not applicable (6) / Refusal (7) / Don't know (8) / No answer (9)

print("\nconstructing CLIMATE CONCERN..")
num_uninformative = clean_df.isnull().sum()['wrclmch']+len(clean_df[clean_df['wrclmch']>=6])
print("Number of people missing or said Not applicable / Refusal / Don't know / No answer: "+str(num_uninformative))

def concern_levels(series):
    if series==1:
        return "Not at all concerned"
    elif series==2:
        return "Not very concerned"
    elif series==3:
        return "Fairly concerned"
    elif series in [4,5]:
        return "Very concerned"
    elif math.isnan(series) or series >= 6:
        return "Other"

clean_df['climate_concern_level'] = clean_df['wrclmch'].apply(concern_levels)


####### ethnicity #######
# Foundation: "ethnicity_bucket" = White / BAME
# ESS: "blgetmg" = Yes (1) / No (2) / Refusal (7) / Don't know (8) / No answer (9)
# (ESS variable is "belong to minority ethnic group")

print("\nconstructing ETHNICITY..")
num_uninformative = clean_df.isnull().sum()['blgetmg']+len(clean_df[clean_df['blgetmg']>=7])
print("Number of people missing or said Refusal / Don't know / No answer: "+str(num_uninformative))


def ethnicity_buckets(series):
    if series==1:
        return "BAME"
    elif series==2:
        return "White"
    else:
        print("unassignable response found - assigning to missing")
        return float("NAN")

clean_df['ethnicity_bucket'] = clean_df['blgetmg'].apply(ethnicity_buckets)

####### urban or rural #######
# Foundation: "urban_rural" = Urban / Rural
# ESS: "domicil" = A big city (1) / Suburbs or outskirts of big city (2) / Town or small city (3) / Country village (4) \
#                  Farm or home in countryside (5) / Refusal (7) / Don't know (8) / No answer (9)

print("\nconstructing URBAN / RURAL..")
num_uninformative = clean_df.isnull().sum()['domicil']+len(clean_df[clean_df['domicil']>=7])
print("Number of people missing or said Refusal / Don't know / No answer: "+str(num_uninformative))


def urban_rural_cats(series):
    if series<=3:
        return "Urban"
    elif series in [4,5]:
        return "Rural"
    else:
        print("unassignable response found - assigning to missing")
        return float("NAN")

clean_df['urban_rural'] = clean_df['domicil'].apply(urban_rural_cats)

####### household size #######
# ESS: "hhmmb" = (continuous)

####### weights #######
# ESS: "spswght"

print("\nexporting to csv...")
# do versions with missings dropped, unnecessary variables dropped
clean_df = clean_df.rename(columns={'pspwght':'weight'})
clean_df = clean_df[['gender','age_bucket','geo_bucket','edu_bucket','climate_concern_level','ethnicity_bucket','urban_rural','weight']]
clean_df = clean_df.dropna()
clean_df.to_csv('data/cleaned_ESS_data_missingsdropped.csv')


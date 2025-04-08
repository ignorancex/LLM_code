#### Function to train ML models to simulations of dynamically-formed black-hole binaries.
#### Copyright: Andrea Antonelli <aantone3@jh.edu>, 19th October 2022.


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import argparse
import pickle


# User inputs
# ---------------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Input parameters')
parser.add_argument('-maxspin' ,'--maxspin', type=float ,metavar=' ',
                    default=0.0, help='Maximum spin sampled from a uniform distribution.')
parser.add_argument('-maxmass' ,'--maxmass', type=float ,metavar=' ',
                    default=9.0, help='Exponent (in power of 10) of the truncation mass for the cluster distribution.')
parser.add_argument('-r' ,'--r', type=float ,metavar=' ',
                    default=1.0, help='Virial radius.')
args = parser.parse_args()


model_name = '_s'+str(args.maxspin)+'_M'+str(args.maxmass)+'_r'+str(args.r)

filename = './data/mergers'+model_name+'.txt'


def df_feature(filename):
    
    
    """
    Function to create a dataset from input simulations of dynamically-formed BHs,
    split it into a random subset that is easier to handle, 
    and wrangle it to create new columns for the overall generation of the binary's BHs.
    """
    
    data= np.loadtxt(filename)
    N, N_features = data.shape
    df = pd.DataFrame(data, columns=['channel', 'a','e', 'm1','m2','chi1','chi2','g1','g2',
                                     'tform','tmerge','zform','zmerge','Nhar','Nsub','q','chieff', 
                                     'theta1', 'theta2', 'dphi','mrem', 'chirem', 'gren', 'vgw', 
                                     'j1', 'j2', 'Mcl_0', 'zcl_form'])
    
    # remove generation of BH above 2nd.
    df = df[df[['g1']].values<=2]
    df = df[df[['g2']].values<=2]
    
    # Take absolute value of channel, forget whether remnant BH is ejected or not.
    df['channel'] = abs(df['channel'])
    
    #Add three new columns.
    N_BHs = len(df.values) # New number of rows.
    df.insert(1, "binary_generation", list(np.arange(0,N_BHs)))
    
    #Create masks for 1g+1g, 1g+2g and 2g+2g events.
    mask1g1g = np.logical_and(df['g1']==1, df['g2']==1)
    mask1g2g = np.logical_or(np.logical_and(df['g1']==1, df['g2']==2),np.logical_and(df['g1']==2, df['g2']==1))
    mask2g2g = np.logical_and(df['g1']==2, df['g2']==2)
    
    # Assign new values.

    df['binary_generation'] = df['binary_generation'].mask(mask1g1g, 1).mask(mask1g2g, 2).mask(mask2g2g,3)
    
    
    # rearrange the masses
    
    df['Mtot']  = df.loc[:,['m1','m2']].sum(axis=1)
    df['logMtot'] = np.log10(df['Mtot'])
    
    df_for_gen = df[['binary_generation', 'm1','m2','chi1','chi2']]
    df_for_formation = df[['channel', 'Mcl_0', 'zcl_form',  'm1','m2','chi1','chi2']]
    
    
    return df_for_gen, df_for_formation



# Specify the dataframes to predict at least one 2g BH, both BHs being 2g, and the formation scenario. 

df_gen = df_feature(filename)[0]
df_form = df_feature(filename)[1]

# Set up the model

clf_RF = RandomForestClassifier(n_estimators = 10,bootstrap = True)



# Train and save the model to predict the binary generation.

y_gen = df_gen["binary_generation"]
X_gen = df_gen.drop("binary_generation", axis=1)
X_train_gen,X_test_gen, y_train_gen,y_test_gen = train_test_split(X_gen,y_gen,test_size=0.3,random_state=0)
    
clf_RF.fit(X_train_gen.values, y_train_gen.values)

filename = 'models/RFclassifier_model'+model_name+'_gen.sav'
pickle.dump(clf_RF, open(filename, 'wb'))



# Train and save model to predict binary's formation scenario.

y_form = df_form["channel"]
X_form = df_form.drop("channel", axis=1)
X_train_form,X_test_form, y_train_form,y_test_form = train_test_split(X_form,y_form,test_size=0.3,random_state=0)

clf_RF.fit(X_train_form.values, y_train_form.values)

filename = 'models/RFclassifier_model'+model_name+'_form.sav'
pickle.dump(clf_RF, open(filename, 'wb'))
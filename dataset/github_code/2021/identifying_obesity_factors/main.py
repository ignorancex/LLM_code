import pysubgroup as ps
import pandas as pd
import csv, re


df = pd.read_csv('one_hot_first_obesity_incident.csv')

# change PT_SEX feature to two different columns
colu = list(df['PT_SEX'])
df.insert(265, "MEN", colu, True)
coluw = []
for each in colu:
    if each == 0:
        coluw.append(1)
    elif each == 1:
        coluw.append(0)
df.insert(266, "WOMEN", coluw, True)

'''
l = df.columns
ll = []
for i in l:
    ll.append(i)
#47 DX_OBESITY
#print (ll)
'''



# In order to run experiment on medication (low-level) data, high-level (visit-level) data should be omitted
high_level_omitted = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 58, 59, 60, 61, 62, 63,
                     69, 70, 82, 88, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
                     208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 229, 245, 246, 247, 249, 251,
                     253, 255, 257, 259, 261, 263, 264]

df = df.drop(df.columns[high_level_omitted], axis=1)

'''
In order to get results of a specific strata, we should uncomment the line below and omit the features (specific columns) that stratum belongs to.
Also, the name of the files should be changed accordingly (instead of "All", name of that strata should be replaced 
'''
#df = df.loc[df['high_income'] == 1]

# in these two lines we seperate the negative from positive data, we set these numbers (as a big number) to include all data. 
data_pos = df.loc[df['is_increased'] == 1].head(60000)
data_neg = df.loc[df['is_increased'] == 0].head(400000)
print(data_neg.shape)
print(data_pos.shape)
data_neg = data_neg.dropna()
data_pos = data_pos.dropna()
print(data_neg.shape)
print(data_pos.shape)
frames = [data_pos, data_neg]
df = pd.concat(frames)
data = df.sample(frac=1).reset_index(drop=True)

target = ps.BinaryTarget('is_increased', 1)

searchspace = ps.create_selectors(data, ignore=['is_increased'])
beam_lengths = [2000, 5000, 10000]
pattern_lengths = [4]
for i in range(len(beam_lengths)):
    for j in range (len(pattern_lengths)):
        task = ps.SubgroupDiscoveryTask(
                        data,
                        target,
                        searchspace,
                        min_quality=0,
                        result_set_size=beam_lengths[i],
                        depth=pattern_lengths[j],
                        # qf=ps.LiftQF())
                        qf=ps.WRAccQF())
                        # qf=ps.LiftQF())

                    #result = ps.GpGrowth().execute(task=task)
        result = ps.BeamSearch().execute(task)
        result_df = result.to_dataframe().to_csv('beamSearch_medication_level_' + str(beam_lengths[i]) + '_'
                                                 + str(pattern_lengths[j])+'WRAccQF_All.csv')
        print ('beam_lengths: ' + str(beam_lengths[i]) + ', pattern_lengths: ' + str(pattern_lengths[j]) +
               ', category: all finished!')


#######################################################
### In order to run experiment on visit (high-level) data, low-level (medication-level) data should be omitted
### this list should be replaced by high_level_omitted list, and the change the "medication" word to "visit" in the
### name of the csv file that is going to save.

low_level_omitted = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 58, 59, 60, 61, 62, 63, 64,
                     65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                     90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                     112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
                     132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
                     152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
                     172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
                     192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211,
                     212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 229, 245, 246, 247, 249, 251, 253, 255, 257, 259,
                     261, 263, 264]

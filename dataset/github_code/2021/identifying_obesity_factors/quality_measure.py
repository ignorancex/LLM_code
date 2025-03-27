import pysubgroup as ps
import pandas as pd
import re, csv



#feature_name = ['MEN', 'WOMEN', 'COMMERCIAL', 'SELF_PAY', 'MEDICARE', 'MEDICAID', 'METRO', 'METRO_ADJUNCENT', 'RURAL', 'LOW_INCOME', 'MED_INCOME', 'HIGH_INCOME',
#                'LATINO', 'AFRICAN_AMERICAN', 'WHITE', 'OTHER_RACE', 'UNDER_THIRTY', 'THIRTIES', 'FOURTIES', 'FIFTIES', 'SIXTIES', 'SEVENTIES']
feature_name = ["All"]

ii = [3, 4, 5]
jj = [2000, 5000, 10000]

#saving feature with corresponding quality
def feature_quality_to_csv (feature_quality_dictionary, i, j, feature_k):
    print(i, j, feature_k)
    with open('avg_quality_0005_medication_level_' + str(j) + '_' + str(i) + '_WRAccQF_om_' + str(feature_k) + '.csv', 'w') as f:
        fieldNames = ['key', 'quality']
        writer = csv.DictWriter(f, fieldnames=fieldNames)
        writer.writeheader()
        for key in feature_quality_dictionary.keys():
            writer.writerow({'key': key, 'quality': feature_quality_dictionary[key]})
        print (j, i, feature_k)

#calculate quality of each feature in rules with WRAcc quality higher than 0.0005         
def calc_quality_count_quality(feature_list, i_value, j_value, feature_k):
    df = pd.read_csv('beamSearch_medication_level_' + str(jj[j]) + '_' + str(ii[i]) + 'WRAccQF_' + str(
        feature_name[kkkkk]) + '.csv')
    feature_quality_dictionary = {}
    for each_feature in feature_list:
        contain_feature = df[df['subgroup'].str.contains(each_feature)]
        feature_quality = sum(list(contain_feature['quality']))
        number_of_SD = len(list(contain_feature['quality']))
        avg_feature_quality = float(feature_quality)/float(number_of_SD)
        feature_quality_dictionary[each_feature] = avg_feature_quality
        #feature_quality_dictionary[each_feature] = float(len(list(contain_feature)))/float(len(df))
    feature_quality_to_csv(feature_quality_dictionary, i_value, j_value, feature_k)


#trying to use some abbreviation for long feature names in the dataset
for kkkkk in range (1):
    for i in range (3):
        for j in range (3):
            df = pd.read_csv('beamSearch_medication_level_' + str(jj[j]) + '_' + str(ii[i]) + 'WRAccQF_'+ str(feature_name[kkkkk]) +'.csv')
            #df = pd.read_csv('updated_obesity_incident_result_' + str(jj[j]) + '_' + str(ii[i]) + '_WRAccQF_om_'+ str(feature_name[kkkkk]) +'.csv')
            df_subgroup = df['subgroup']
            for m in range(len(df_subgroup)):
                # print (df_subgroup[ii])
                df_subgroup[m] = df_subgroup[m].replace(
                    'Antidepressant - Selective Serotonin Reuptake Inhibitors (SSRIs)', 'SSRIs')
                df_subgroup[m] = df_subgroup[m].replace(
                    'Antidepressant-Norepinephrine and Dopamine Reuptake Inhibitors (NDRIs)', 'NDRIs')
                df_subgroup[m] = df_subgroup[m].replace(
                    'Antidepressant - Serotonin-Norepinephrine Reuptake Inhibitors (SNRIs)', 'SNRIs')
                df_subgroup[m] = df_subgroup[m].replace(
                    'Antidepressant-Norepinephrine and Dopamine Reuptake Inhibitors (NDRIs)', 'NDRIs')
                df_subgroup[m] = df_subgroup[m].replace('Antihyperlipidemic - HMG CoA Reductase Inhibitors (statins)',
                                                        'statins')
                df_subgroup[m] = df_subgroup[m].replace(
                    'Antidepressant - Serotonin-2 Antagonist-Reuptake Inhibitors (SARIs)', 'SARIs')
                df_subgroup[m] = df_subgroup[m].replace(
                    'Antidepressant-Tricyclics and Related (Non-Select Reuptake Inhibitors)', 'NSRIs')
                df_subgroup[m] = df_subgroup[m].replace('Antidepressant - Alpha-2 Receptor Antagonists (NaSSA)',
                                                        'NaSSA')
                df_subgroup[m] = df_subgroup[m].replace('Antihyperglycemic - Insulin Response Enhancers', 'IREs')
                df_subgroup[m] = df_subgroup[m].replace('ACE Inhibitors (ACEIs) Combinations', 'ACEIs')
                df_subgroup[m] = df_subgroup[m].replace('Angiotensin II Receptor Blocker (ARBs) and ARB Combinations',
                                                        'ARBs Combinators')
                df_subgroup[m] = df_subgroup[m].replace('Angiotensin II Receptor Blocker (ARBs)', 'ARBs')
                df_subgroup[m] = df_subgroup[m].replace('Angiotensin II Receptor Blockers (ARBs)', 'ARBs')
                df_subgroup[m] = df_subgroup[m].replace('Angiotensin II Receptor Blocker (ARB)-Diuretic Combinations',
                                                        'ARBs Diuretic')
                df_subgroup[m] = df_subgroup[m].replace('ACE Inhibitor and Diuretic Combinations', 'ACE Diuretic Comb')
                df_subgroup[m] = df_subgroup[m].replace('Antihyperglycemic (Antidiabetic) Combinations', 'Antidiabetic')

            df['subgroup'] = df_subgroup

            df = df[df.quality > 0.0005]


            dfh_sg = df['subgroup']
            dfh_sg_list = list(dfh_sg)
            feature_list = []
            for iii in dfh_sg_list:
                sg = iii.split('AND')
                for jjj in sg:
                    jjj = jjj.strip()
                    jjj = re.sub('==1.0', '', jjj)
                    jjj = re.sub('==1', '', jjj)
                    feature_list.append(jjj)
            feature_list = list(set(feature_list))
            calc_quality_count_quality(feature_list, ii[i], jj[j], feature_name[kkkkk])



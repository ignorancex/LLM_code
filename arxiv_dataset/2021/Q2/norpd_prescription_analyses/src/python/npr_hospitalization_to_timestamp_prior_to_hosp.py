

import numpy as np
from pyspark.sql.functions import collect_list, struct

#We impose a limitation that the previous hospitalization must be more than "days_prior"
#days before the timestamp we collect
def collect_df_with_list_of_hospitalizations_from_npr(npr_df):
    spark_hospitalization_list = npr.groupBy("id")\
    .agg(collect_list(struct("diffdager_inn", "diffdager_ut")))
    
    # Get two fields: ID and hospitalization list created from aggregate above
    col_labels = spark_hospitalization_list.schema.names
    # Rename second field to 'hospitalization_list' for readability and ease of use
    spark_hospitalization_list = spark_hospitalization_list\
    .select(col(col_labels[0]),col(col_labels[1]).alias("hospitalization_list"))
    
    return df


def load_df_with_hosp_to_pandas_and_sort_chronologically(singleListOfHospitalizations):
    # Sort chronologically
    
    lhosp = singleListOfHospitalizations
    sorter = lambda l: sorted(l, key=lambda tup: tup[1])
    lhosp[lhosp.columns[1]] = lhosp[lhosp.columns[1]].apply(sorter)
    return lhosp

#We impose a limitation that the previous hospitalization must be more than "days_prior"
#days before the timestamp we collect

def get_timestamp_from_n_longest_windows(hosp_list, 
                                         n=1, 
                                         days_prior=30, 
                                         max_sample_per_patient=3):
    tstamps = []
    i=0
    #-270 is the hard limit for dates in the hospitalized dataset
    #pre_first_hosp_end is the timestamp of the first hospitalization of the patient
    #Thus we obtain min duration without hospitalization as ..
    # diff(-270, first_hospitalization)
    pre_first_hosp_start = -270
    pre_first_hosp_end = hosp_list[i][0]
    duration = pre_first_hosp_end-pre_first_hosp_start
    if duration > 2*days_prior:
        tstamps.append([pre_first_hosp_end-days_prior, duration])
        
    while i<(len(hosp_list) - 1):
        end = hosp_list[i+1][0]
        start = hosp_list[i][1]
        duration = end-start
        tstamp = end-days_prior
        tstamps.append([tstamp, duration])
        i+=1
    return list(
                map(lambda x:[x[0]],
                    filter(lambda x: x[1]>days_prior,
                           #Sort and take only the N longest windows per patient
                           sorted(tstamps, key=lambda x: x[1], reverse=True)[:n])
                   )
               )
from pyspark.sql.functions import collect_list, struct, col, explode, rand


def lookup_drug(active_drug_table, patient_id, timestamp):
    cond1 = col("id") == patient_id
    
    cond2_1 = col("treatment_start") <= timestamp
    
    cond2_2 = col("treatment_end") >= timestamp
    return active_drug_table.where(cond1) \
        .where(cond2_1 & cond2_2)

# We may need some random number assigned with each patient ID in patientsDF
# Reason: If we sample with replacement we may get the same patient multiple times
#     Thus, to ensure that the resulting lists contain
def lookup_drugs(active_drug_table, patientsDF):
    cond1_1 = col("dd.id") == col("patdf.id")
    cond1_2 = col("patdf.timestamp") >= col("dd.treatment_start")
    cond1_3 = col("patdf.timestamp") <= col("dd.treatment_end")
    dd = active_drug_table.alias("dd")
    return dd \
        .join(patientsDF.alias("patdf")) \
        .where(
            cond1_1
            &
            cond1_2
            &
            cond1_3
          )\
        .select("dd.*", "patdf.timestamp")\
        .groupBy("id", "timestamp") \
        .agg(collect_list(col("drugcode")).alias("active_drugs")) \


def sample_from_healthy_population(df, frac, withreplacement = True):
    return df.sample(withreplacement, frac) \
        .select(
                col("id")
                ,(rand()*365).alias("timestamp").cast("int")
               )

def sample_from_hosp_population(df, frac, withreplacement = True):
    return df.sample(withreplacement, frac)


def sample_and_get_drugs_from_both(
        dfhosp
        ,dfheal
        ,active_drug_table
        ,frac_hosp
        ,frac_heal
    ):
    hosp_pat = sample_from_hosp_population(dfhosp, frac_hosp)
    hosp_active_drugs = lookup_drugs(active_drug_table, hosp_pat)
    
    heal_pat = sample_from_healthy_population(dfheal, frac_heal)
    heal_active_drugs = lookup_drugs(active_drug_table, heal_pat)
    return (hosp_active_drugs, heal_active_drugs)

    
           
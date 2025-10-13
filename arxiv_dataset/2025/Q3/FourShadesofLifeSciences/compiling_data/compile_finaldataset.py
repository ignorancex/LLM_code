#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "compilation of data set for FSoLS dataset"
"the amount of items per class and data source needs to be stated manually"
(
    "composing ratio refers to 14 topics and several data sources. Please add the numbers for each topic and database. "
    "PMC = PubMed Central"
    "CMT = CompMedTherapies "
    "goethe" = "School of Spiritual Science Medical Section at the Goetheanum"
    "WebMD = WebMD "
    "HHP  = HarvardMedicalSchool "
    "MH = MensHealth "
    "WH = WomensHealth "
    "MP = MedlinePlus"
    "JEBIM = sage journal of evidence based integrative medicine = "
    "NN = Naturals news"
    "HIN = Health impact news"
    "health DOT news"
    "infowars"
    "MCL = Mercolas Censored Library "
    "homeoJ = Homeopathic Journal"
    "IJRH = Indian Journal for Research Homeopathy"
)
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2024-2025 by Eva Seidlmayer"
__license__ = "ISC license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "FSoLS 2024-2025 "


import pandas as pd
import numpy as np
import re
import argparse

SPLIT_RATIO = 0.15


def load_dataset_sci(base_url, col_list):
    # scientific
    pmc_df = pd.read_csv(
        base_url
        + "scientific_PMC_cleaned.csv",
        usecols=col_list,
    )
    scientific_df = pmc_df.drop_duplicates()
    scientific_df.loc[:, "text"] = scientific_df["text"].str.strip()
    patterns = [
        r"^\s*BACKGROUND",
        r"^\s*Background",
        r"Background",
        r"BACKGROUND",
        r"^\s*Highlights",
        r"^\s*Objective",
        r"^\s*Abstract",
        r"ABSTRACT",
        r"^\s*Purpose",
        r"^\s*Objectives",
        r"^\s*OBJECTIVE",
        r"^\s*Summary",
        r"SUMMARY",
        r"Introduction",
        r"Aims",
        r"^\s*Key Points",
        r"Rationale",
        r"Context",
        r"^\s*ObjectiveBackground",
        "PURPOSE",
    ]

    for pattern in patterns:
        scientific_df["text"] = scientific_df["text"].str.replace(
            pattern, "", regex=True
        )
    scientific_df.to_csv(
        base_url + "complete_scientific.csv", index=False
    )
    scientific_df_path = base_url + "complete_scientific.csv"
    return scientific_df_path


def load_dataset_vern(base_url, col_list):
    #vernacular science
    webmd_df = pd.read_csv(
        base_url
        + "vernacular_WebMed_cleaned.csv",
        usecols=col_list,
    ).drop_duplicates()
    
    hhp_df = pd.read_csv(
        base_url
        + "vernacular_HHP_cleaned.csv",
        usecols=col_list,
    ).drop_duplicates()

    MH_df = pd.read_csv(
        base_url
        + "vernacular_MH_cleaned.csv",
        usecols=col_list,
    ).drop_duplicates()

    WH_df = pd.read_csv(
        base_url
        + "vernacular_WH_cleaned.csv",
        usecols=col_list,
    ).drop_duplicates()

    MP_df = pd.read_csv(
        base_url
        + "vernacular_MedlinePlus_cleaned.csv",
        usecols=col_list,
    ).drop_duplicates()

    Mayo_df = pd.read_csv(
        base_url
        + "vernacular_mayoclinic_cleaned.csv",
        usecols=col_list,
    ).drop_duplicates()

    vernacular_df_list = [
        webmd_df,
        hhp_df,
        MH_df,
        WH_df,
        MP_df,
        Mayo_df,
    ]
    print("loaded vernacular data")
    print("webmd_df len", len(webmd_df))
    print("hhp_df len", len(hhp_df))
    print("mens health_df len", len(MH_df))
    print("womens df len", len(WH_df))
    print("Medline Plus_df len", len(MP_df))
    print("Mayo_df len", len(Mayo_df))

    vernacular_df = pd.concat(vernacular_df_list, axis=0)
    vernacular_df["text"] = vernacular_df["text"].str.strip()

    vernacular_df["text"] = vernacular_df["text"].apply(
        lambda text: re.sub(
            r".*\b(editor|contributor)\b", "", text, flags=re.IGNORECASE
        ).strip()
    )
    vernacular_df.to_csv(base_url + "complete_vernacular.csv", index=False)
    vernacular_df_path = base_url + "complete_vernacular.csv"
    return vernacular_df_path


def load_dataset_alt(base_url, col_list):
    ## alternative science
    JBIM_df = pd.read_csv(
        base_url
        + "alternative_JEBIM_cleaned.csv",
        usecols=col_list,
    )
    
    CMT_df = pd.read_csv(
        base_url
        + "alternative_CMT_cleaned.csv",
        usecols=col_list,
    )
    
    HomeoJ_df = pd.read_csv(
        base_url
        + "alternative_homeoJ_cleaned.csv",
        usecols=col_list,
    )
    
    IJRH_df = pd.read_csv(
        base_url
        + "alternative_IJRH_cleaned.csv",
        usecols=col_list,
    )
    
    Goethe_df = pd.read_csv(
        base_url
        + "alternative_goethe_cleaned.csv",
        usecols=col_list,
    )
    print("loaded alt scienctific data")
    print("JBIM_df len",  len(JBIM_df))
    print("CMT_df len", len(CMT_df))
    print("HomeoJ_df len", len(HomeoJ_df))
    print("IJRH_df len", len(IJRH_df))
    print("Goethe_df len", len(Goethe_df))

    alt_df_list = [
        JBIM_df,
        CMT_df,
        HomeoJ_df,
        IJRH_df,
        Goethe_df,
    ]
    alternative_df = pd.concat(alt_df_list, axis=0, ignore_index=True).drop_duplicates()
    alternative_df["text"] = alternative_df["text"].str.split(
        r"licensee biomed", expand=True
    )[0]
    alternative_df["text"] = alternative_df["text"].str.split(
        r"submit your next manuscript to BioMed", expand=True
    )[0]
    alternative_df["text"] = alternative_df["text"].str.split(
        r"licensee BioMed Central Ltd.", expand=True
    )[0]
    patterns = [
        (r"\bThe Author s\b.*?\.(?!\d)", ""),
        (r"\stion", "tion"),
        ("doi.org", ""),
        (r"\sing", "ing"),
        (" ing", "ing"),
        (" sion", "sion"),
        (r"\ssion", "sion"),
        (r"\sment", "ment"),
        (r"nethics", "ethics"),
        (r"ndisease", "disease"),
        (r"nconsent", "consent"),
        (r"nabbreviations", "abbreviations"),
        ("creativecommons.org", ""),
        ("licence", ""),
        ("commons", ""),
        ("manuscript", ""),
        ("keywords", ""),
        ("screeing", "screening"),
        ("maagement", "management"),
        ("inhibintor", "inhibitor"),
        ("nbrain", "brain"),
        ("nfunding", "funding"),
        ("4.0", ""),
        (r"signifi\scant", "significant"),
        ("nyes", "yes"),
        (
            "publicdomain zero 1.0   applies to the data made available in this article  unless otherwise stated in a credit line to the data.",
            "",
        ),
        (
            "publicdomain zero 1.0   applies to the data made available in this article  nunless otherwise stated in a credit line to the data.",
            "",
        ),
        ("The Creative  Public Domain Dedication waiver", ""),
        ("  ntive", "tive"),
        ("navailability", "availability"),
        ("nmm", "mm"),
        ("metioned", "mentioned"),
        ("signifi ", "siginificant"),
        ("nchinese", "chinese"),
        ("sagepub.comjournalspermissions.nav", ""),
        ("attetion", "attention"),
        ("intervetion", "intervention"),
    ]
    for pattern, replacement in patterns:
        alternative_df["text"] = alternative_df["text"].str.replace(
            pattern, replacement, regex=True, flags=re.IGNORECASE
        )
    alternative_df.to_csv(base_url + "complete_alternative", index=False)
    alternative_df_path = base_url + "complete_alternative.csv"
    return alternative_df_path


def load_dataset_dis(base_url, col_list):
    ## disinformation
    NN_df = pd.read_csv(
        base_url
        + "disinfo_NaturalNews_cleaned.csv",
        usecols=col_list,
    )
    
    HIN_df = pd.read_csv(
        base_url
        + "disinfo_healthimpactnews_cleaned.csv",
        usecols=col_list,
    )
    Mercola_df = pd.read_csv(
        base_url
        + "disinfo_mercola_cleaned.csv",
        usecols=col_list,
    )
    
    HN_df = pd.read_csv(
        base_url
        + "disinfo_healthDOTnews_cleaned.csv",
        usecols=col_list,
    )
    
    IW_df = pd.read_csv(
        base_url
        + "disinfo_infowars_cleaned.csv",
        usecols=col_list,
    )
    

    print("loaded disinfo data")
    print("NN_df len", len(NN_df))
    print("HIN_df len", len(HIN_df))
    print("Mercola_df len", len(Mercola_df))
    print("HN_df len", len(HN_df))
    print("IW_df len", len(IW_df))

    dis_list = [
        NN_df,
        HIN_df,
        Mercola_df,
        HN_df,
        IW_df,
    ]
    disinfo_df = pd.concat(dis_list, axis=0, ignore_index=True).drop_duplicates()

    patterns = [
        ("mercola", ""),
        (r"you\\", "you "),
        (r"it\\", "it "),
        (r"don\\\'t", "don't"),
        (r"alzheimer\\", "alzheimer"),
        ("marked.the", "marked. the"),
    ]
    for pattern, replacement in patterns:
        disinfo_df["text"] = disinfo_df["text"].str.replace(
            pattern, replacement, regex=True, flags=re.IGNORECASE
        )
    disinfo_df.to_csv(base_url + "complete_disinfo_2025-04-01.csv", index=False)
    disinfo_df_path = base_url + "complete_disinfo_2025-04-01.csv"
    return disinfo_df_path


def filter_vernacular(
    vernacular_df,
    topic,
    number_WebMD,
    number_HHP,
    number_MH,
    number_WH,
    number_MP,
    number_Mayo,
):
    pop_df = vernacular_df.loc[vernacular_df["tags"] == f"{topic}"]
    pop_df_WebMD = pop_df.loc[pop_df["data-source"] == "WebMD"].head(number_WebMD)
    pop_df_HHP = pop_df.loc[pop_df["data-source"] == "HarvardMedicalSchool"].head(
        number_HHP
    )
    pop_df_MH = pop_df.loc[pop_df["data-source"] == "MensHealth"].head(number_MH)
    pop_df_WH = pop_df.loc[pop_df["data-source"] == "WomensHealth"].head(number_WH)
    pop_df_MP = pop_df.loc[pop_df["data-source"] == "MedlinePlus"].head(number_MP)
    pop_df_Mayo = pop_df.loc[pop_df["data-source"] == "mayoclinic"].head(number_Mayo)
    filtered_vernacular_df = pd.concat(
        [pop_df_WebMD, pop_df_HHP, pop_df_MH, pop_df_WH, pop_df_MP, pop_df_Mayo],
        axis=0,
        ignore_index=True,
    )

    return filtered_vernacular_df


def filter_alternative(
    alternative_df,
    topic,
    number_JEBIM,
    number_CMT,
    number_goethe,
    number_homeoj,
    number_ijrh,
):
    alt_df = alternative_df.loc[alternative_df["tags"] == f"{topic}"]
    print("YYY", alt_df)
    alt_df_JEBIM = alt_df.loc[alt_df["data-source"] == "JEBIM"].head(number_JEBIM)
    alt_df_CMT = alt_df.loc[alt_df["data-source"] == "CompMedTherapies"].head(
        number_CMT
    )
    alt_df_Goethe = alt_df.loc[alt_df["data-source"] == "PAAM/Goetheaneum-list"].head(
        number_goethe
    )
    alt_df_HomeoJ = alt_df.loc[alt_df["data-source"] == "homeopathicjournal"].head(
        number_homeoj
    )
    alt_df_IJRH = alt_df.loc[
        alt_df["data-source"] == "Indian-research-Homeopathy"
    ].head(number_ijrh)
    filtered_alternative_df = pd.concat(
        [alt_df_JEBIM, alt_df_CMT, alt_df_Goethe, alt_df_HomeoJ, alt_df_IJRH],
        axis=0,
        ignore_index=True,
    )
    print("XXXXXXXXXXXXXXXXXXXXXXX", filtered_alternative_df)
    return filtered_alternative_df


def filter_disinformation(
    disinfo_df, topic, number_NN, number_HIN, number_MCL, number_HN, number_IW
):
    disi_df = disinfo_df.loc[disinfo_df["tags"] == f"{topic}"]
    disi_df_NN = disi_df.loc[disi_df["data-source"] == "NaturalNews"].head(number_NN)
    disi_df_HIN = disi_df.loc[disi_df["data-source"] == "HealthImpactNews"].head(
        number_HIN
    )
    disi_df_MCL = disi_df.loc[disi_df["data-source"] == "Mercola"].head(number_MCL)
    disi_df_HN = disi_df.loc[disi_df["data-source"] == "HealthDOTNews"].head(number_HN)
    disi_df_IW = disi_df.loc[disi_df["data-source"] == "InfoWars"].head(number_IW)
    filtered_disinform_df = pd.concat(
        [disi_df_NN, disi_df_HIN, disi_df_MCL, disi_df_HN, disi_df_IW],
        axis=0,
        ignore_index=True,
    )
    return filtered_disinform_df


def filter_for_cumin(scientific_df, vernacular_df, alternative_df, disinfo_df):
    topic = "cumin"

    ## scientific
    filtered_sci_df = scientific_df[
        scientific_df["tags"].str.contains("cumin", case=False, na=False)
    ].head(7)
    split_ratio = calc_split_ratio(filtered_sci_df)
    print("split_ratio urine", split_ratio)
    filtered_sci_df_train, filtered_sci_df_test = split_train_test_data(
        filtered_sci_df, split_ratio
    )
    ## popular
    number_WebMD = 4
    number_HHP = 0
    number_MH = 2
    number_WH = 0
    number_MP = 0
    number_Mayo = 1
    filtered_pop_df = filter_vernacular(
        vernacular_df,
        topic,
        number_WebMD,
        number_HHP,
        number_MH,
        number_WH,
        number_MP,
        number_Mayo,
    )
    filtered_pop_df_train, filtered_pop_df_test = split_train_test_data(
        filtered_pop_df, split_ratio
    )

    ## alternative science
    number_JEBIM = 4
    number_CMT = 3
    number_goethe = 0
    number_homeoj = 0
    number_ijrh = 0
    filtered_alt_df = filter_alternative(
        alternative_df,
        topic,
        number_JEBIM,
        number_CMT,
        number_goethe,
        number_homeoj,
        number_ijrh,
    )
    filtered_alt_df_train, filtered_alt_df_test = split_train_test_data(
        filtered_alt_df, split_ratio
    )

    ## disinformation
    number_NN = 3
    number_HIN = 3
    number_MCL = 1
    number_HN = 0
    number_IW = 0
    filtered_dis_df = filter_disinformation(
        disinfo_df, topic, number_NN, number_HIN, number_MCL, number_HN, number_IW
    )
    filtered_dis_df_train, filtered_dis_df_test = split_train_test_data(
        filtered_dis_df, split_ratio
    )

    all_cumin_train = pd.concat(
        [
            filtered_sci_df_train,
            filtered_pop_df_train,
            filtered_alt_df_train,
            filtered_dis_df_train,
        ],
        axis=0,
        ignore_index=True,
    )
    all_cumin_test = pd.concat(
        [
            filtered_sci_df_test,
            filtered_pop_df_test,
            filtered_alt_df_test,
            filtered_dis_df_test,
        ],
        axis=0,
        ignore_index=True,
    )

    return all_cumin_train, all_cumin_test


def filter_for_dementia(scientific_df, vernacular_df, alternative_df, disinfo_df):
    topic = "dementia"

    ## scientific
    # filtered_scientific_df = scientific_df.loc[scientific_df['tags']=='dementia']
    filtered_scientific_df = scientific_df[
        scientific_df["tags"].str.contains("dementia", case=False, na=False)
    ].head(79)
    split_ratio = calc_split_ratio(filtered_scientific_df)
    print("split_ratio dementia", split_ratio)
    filtered_sci_df_train, filtered_sci_df_test = split_train_test_data(
        filtered_scientific_df, split_ratio
    )

    ## popular
    number_WebMD = 27
    number_HHP = 22
    number_MH = 5
    number_WH = 6
    number_MP = 19
    number_Mayo = 0
    filtered_pop_df = filter_vernacular(
        vernacular_df,
        topic,
        number_WebMD,
        number_HHP,
        number_MH,
        number_WH,
        number_MP,
        number_Mayo,
    )
    filtered_pop_df_train, filtered_pop_df_test = split_train_test_data(
        filtered_pop_df, split_ratio
    )

    ## alternative science
    number_JEBIM = 30
    number_CMT = 31
    number_homeoj = 12
    number_goethe = 1
    number_ijrh = 5
    filtered_alt_df = filter_alternative(
        alternative_df,
        topic,
        number_JEBIM,
        number_CMT,
        number_goethe,
        number_homeoj,
        number_ijrh,
    )
    filtered_alt_df_train, filtered_alt_df_test = split_train_test_data(
        filtered_alt_df, split_ratio
    )

    ## disinformation
    number_NN = 8
    number_HIN = 49
    number_MCL = 17
    number_HN = 5
    number_IW = 1
    filtered_dis_df = filter_disinformation(
        disinfo_df, topic, number_NN, number_HIN, number_MCL, number_HN, number_IW
    )
    filtered_dis_df_train, filtered_dis_df_test = split_train_test_data(
        filtered_dis_df, split_ratio
    )

    all_dementia_train = pd.concat(
        [
            filtered_sci_df_train,
            filtered_pop_df_train,
            filtered_alt_df_train,
            filtered_dis_df_train,
        ],
        axis=0,
        ignore_index=True,
    )

    all_dementia_test = pd.concat(
        [
            filtered_sci_df_test,
            filtered_pop_df_test,
            filtered_alt_df_test,
            filtered_dis_df_test,
        ],
        axis=0,
        ignore_index=True,
    )
    return all_dementia_train, all_dementia_test


def filter_for_heartattack(scientific_df, vernacular_df, alternative_df, disinfo_df):
    topic = "heartattack"

    ## scientific
    filtered_scientific_df = scientific_df.loc[
        scientific_df["tags"] == "heartattack"
    ].head(48)
    split_ratio = calc_split_ratio(filtered_scientific_df)
    print("split_ratio heartattack", split_ratio)
    filtered_sci_df_train, filtered_sci_df_test = split_train_test_data(
        filtered_scientific_df, split_ratio
    )

    ## popular
    number_WebMD = 11
    number_HHP = 6
    number_MH = 12
    number_WH = 10
    number_MP = 9
    number_Mayo = 0
    filtered_pop_df = filter_vernacular(
        vernacular_df,
        topic,
        number_WebMD,
        number_HHP,
        number_MH,
        number_WH,
        number_MP,
        number_Mayo,
    )
    filtered_pop_df_train, filtered_pop_df_test = split_train_test_data(
        filtered_pop_df, split_ratio
    )

    ## alternative science
    number_JEBIM = 45
    number_CMT = 45
    number_homeoj = 3
    number_goethe = 0
    number_ijrh = 0
    filtered_alt_df = filter_alternative(
        alternative_df,
        topic,
        number_JEBIM,
        number_CMT,
        number_goethe,
        number_homeoj,
        number_ijrh,
    )
    filtered_alt_df_train, filtered_alt_df_test = split_train_test_data(
        filtered_alt_df, split_ratio
    )

    ## disinformation
    number_NN = 13
    number_HIN = 20
    number_MCL = 13
    number_HN = 2
    number_IW = 0
    filtered_dis_df = filter_disinformation(
        disinfo_df, topic, number_NN, number_HIN, number_MCL, number_HN, number_IW
    )
    filtered_dis_df_train, filtered_dis_df_test = split_train_test_data(
        filtered_dis_df, split_ratio
    )

    all_heartattack_df_train = pd.concat(
        [
            filtered_sci_df_train,
            filtered_pop_df_train,
            filtered_alt_df_train,
            filtered_dis_df_train,
        ],
        axis=0,
        ignore_index=True,
    )
    all_heartattack_df_test = pd.concat(
        [
            filtered_sci_df_test,
            filtered_pop_df_test,
            filtered_alt_df_test,
            filtered_dis_df_test,
        ],
        axis=0,
        ignore_index=True,
    )

    return all_heartattack_df_train, all_heartattack_df_test


def filter_for_insomnia(scientific_df, vernacular_df, alternative_df, disinfo_df):
    topic = "insomnia"

    ## scientific
    filtered_scientific_df = scientific_df.loc[
        scientific_df["tags"] == "insomnia"
    ].head(41)
    split_ratio = calc_split_ratio(filtered_scientific_df)
    print("split_ratio insomnia", split_ratio)
    filtered_sci_df_train, filtered_sci_df_test = split_train_test_data(
        filtered_scientific_df, split_ratio
    )

    ## popular
    number_WebMD = 19
    number_HHP = 11
    number_MH = 0
    number_WH = 6
    number_MP = 5
    number_Mayo = 0
    filtered_pop_df = filter_vernacular(
        vernacular_df,
        topic,
        number_WebMD,
        number_HHP,
        number_MH,
        number_WH,
        number_MP,
        number_Mayo,
    )
    filtered_pop_df_train, filtered_pop_df_test = split_train_test_data(
        filtered_pop_df, split_ratio
    )

    ## alternative science
    number_JEBIM = 9
    number_CMT = 9
    number_homeoj = 13
    number_goethe = 3
    number_ijrh = 7
    filtered_alt_df = filter_alternative(
        alternative_df,
        topic,
        number_JEBIM,
        number_CMT,
        number_goethe,
        number_homeoj,
        number_ijrh,
    )
    filtered_alt_df_train, filtered_alt_df_test = split_train_test_data(
        filtered_alt_df, split_ratio
    )

    ## disinformation
    number_NN = 32
    number_HIN = 0
    number_MCL = 16
    number_HN = 0
    number_IW = 0
    filtered_dis_df = filter_disinformation(
        disinfo_df, topic, number_NN, number_HIN, number_MCL, number_HN, number_IW
    )
    filtered_dis_df_train, filtered_dis_df_test = split_train_test_data(
        filtered_dis_df, split_ratio
    )

    all_insomnia_df_train = pd.concat(
        [
            filtered_sci_df_train,
            filtered_pop_df_train,
            filtered_alt_df_train,
            filtered_dis_df_train,
        ],
        axis=0,
        ignore_index=True,
    )
    all_insomnia_df_test = pd.concat(
        [
            filtered_sci_df_test,
            filtered_pop_df_test,
            filtered_alt_df_test,
            filtered_dis_df_test,
        ],
        axis=0,
        ignore_index=True,
    )
    return all_insomnia_df_train, all_insomnia_df_test


def filter_for_menopause(scientific_df, vernacular_df, alternative_df, disinfo_df):
    topic = "menopause"

    ## scientific
    filtered_scientific_df = scientific_df.loc[
        scientific_df["tags"] == "menopause"
    ].head(37)
    split_ratio = calc_split_ratio(filtered_scientific_df)
    print("split_ratio menopause", split_ratio)
    filtered_sci_df_train, filtered_sci_df_test = split_train_test_data(
        filtered_scientific_df, split_ratio
    )

    ## popular
    number_WebMD = 19
    number_HHP = 9
    number_MH = 0
    number_WH = 5
    number_MP = 4
    number_Mayo = 0
    filtered_pop_df = filter_vernacular(
        vernacular_df,
        topic,
        number_WebMD,
        number_HHP,
        number_MH,
        number_WH,
        number_MP,
        number_Mayo,
    )
    filtered_pop_df_train, filtered_pop_df_test = split_train_test_data(
        filtered_pop_df, split_ratio
    )

    ## alternative science
    number_JEBIM = 10
    number_CMT = 10
    number_homeoj = 9
    number_goethe = 2
    number_ijrh = 6
    filtered_alt_df = filter_alternative(
        alternative_df,
        topic,
        number_JEBIM,
        number_CMT,
        number_goethe,
        number_homeoj,
        number_ijrh,
    )
    filtered_alt_df_train, filtered_alt_df_test = split_train_test_data(
        filtered_alt_df, split_ratio
    )

    ## disinformation
    number_NN = 10
    number_HIN = 21
    number_MCL = 13
    number_HN = 2
    number_IW = 1
    filtered_dis_df = filter_disinformation(
        disinfo_df, topic, number_NN, number_HIN, number_MCL, number_HN, number_IW
    )
    filtered_dis_df_train, filtered_dis_df_test = split_train_test_data(
        filtered_dis_df, split_ratio
    )

    all_menopause_df_train = pd.concat(
        [
            filtered_sci_df_train,
            filtered_pop_df_train,
            filtered_alt_df_train,
            filtered_dis_df_train,
        ],
        axis=0,
        ignore_index=True,
    )
    all_menopause_df_test = pd.concat(
        [
            filtered_sci_df_test,
            filtered_pop_df_test,
            filtered_alt_df_test,
            filtered_dis_df_test,
        ],
        axis=0,
        ignore_index=True,
    )
    return all_menopause_df_train, all_menopause_df_test


def filter_for_stroke(scientific_df, vernacular_df, alternative_df, disinfo_df):
    topic = "stroke"

    ## scientific
    filtered_scientific_df = scientific_df.loc[scientific_df["tags"] == "stroke"].head(
        91
    )
    split_ratio = calc_split_ratio(filtered_scientific_df)
    print("split_ratio stroke", split_ratio)
    filtered_sci_df_train, filtered_sci_df_test = split_train_test_data(
        filtered_scientific_df, split_ratio
    )

    ## popular
    number_WebMD = 117
    number_HHP = 1
    number_MH = 6
    number_WH = 3
    number_MP = 22
    number_Mayo = 0
    filtered_pop_df = filter_vernacular(
        vernacular_df,
        topic,
        number_WebMD,
        number_HHP,
        number_MH,
        number_WH,
        number_MP,
        number_Mayo,
    )
    filtered_pop_df_train, filtered_pop_df_test = split_train_test_data(
        filtered_pop_df, split_ratio
    )

    ## alternative science
    number_JEBIM = 26
    number_CMT = 26
    number_homeoj = 28
    number_goethe = 0
    number_ijrh = 12
    filtered_alt_df = filter_alternative(
        alternative_df,
        topic,
        number_JEBIM,
        number_CMT,
        number_goethe,
        number_homeoj,
        number_ijrh,
    )
    filtered_alt_df_train, filtered_alt_df_test = split_train_test_data(
        filtered_alt_df, split_ratio
    )

    ## disinformation
    number_NN = 30
    number_HIN = 23
    number_MCL = 31
    number_HN = 5
    number_IW = 1
    filtered_dis_df = filter_disinformation(
        disinfo_df, topic, number_NN, number_HIN, number_MCL, number_HN, number_IW
    )
    filtered_dis_df_train, filtered_dis_df_test = split_train_test_data(
        filtered_dis_df, split_ratio
    )

    all_stroke_df_train = pd.concat(
        [
            filtered_sci_df_train,
            filtered_pop_df_train,
            filtered_alt_df_train,
            filtered_dis_df_train,
        ],
        axis=0,
        ignore_index=True,
    )
    all_stroke_df_test = pd.concat(
        [
            filtered_sci_df_test,
            filtered_pop_df_test,
            filtered_alt_df_test,
            filtered_dis_df_test,
        ],
        axis=0,
        ignore_index=True,
    )
    return all_stroke_df_train, all_stroke_df_test


def filter_for_tobacco(scientific_df, vernacular_df, alternative_df, disinfo_df):
    topic = "tobacco"

    ## scientific
    filtered_scientific_df = scientific_df.loc[scientific_df["tags"] == "tobacco"].head(
        21
    )
    split_ratio = calc_split_ratio(filtered_scientific_df)
    print("split_ratio tobacco", split_ratio)
    filtered_sci_df_train, filtered_sci_df_test = split_train_test_data(
        filtered_scientific_df, split_ratio
    )

    ## popular
    number_WebMD = 0
    number_HHP = 1
    number_MH = 6
    number_WH = 1
    number_MP = 5
    number_Mayo = 8
    filtered_pop_df = filter_vernacular(
        vernacular_df,
        topic,
        number_WebMD,
        number_HHP,
        number_MH,
        number_WH,
        number_MP,
        number_Mayo,
    )
    filtered_pop_df_train, filtered_pop_df_test = split_train_test_data(
        filtered_pop_df, split_ratio
    )

    ## alternative science
    number_JEBIM = 0
    number_CMT = 0
    number_homeoj = 14
    number_goethe = 1
    number_ijrh = 6
    filtered_alt_df = filter_alternative(
        alternative_df,
        topic,
        number_JEBIM,
        number_CMT,
        number_goethe,
        number_homeoj,
        number_ijrh,
    )
    filtered_alt_df_train, filtered_alt_df_test = split_train_test_data(
        filtered_alt_df, split_ratio
    )

    ## disinformation
    number_NN = 10
    number_HIN = 3
    number_MCL = 7
    number_HN = 1
    number_IW = 0
    filtered_dis_df = filter_disinformation(
        disinfo_df, topic, number_NN, number_HIN, number_MCL, number_HN, number_IW
    )
    filtered_dis_df_train, filtered_dis_df_test = split_train_test_data(
        filtered_dis_df, split_ratio
    )

    all_tobacco_df_train = pd.concat(
        [
            filtered_sci_df_train,
            filtered_pop_df_train,
            filtered_alt_df_train,
            filtered_dis_df_train,
        ],
        axis=0,
        ignore_index=True,
    )
    all_tobacco_df_test = pd.concat(
        [
            filtered_sci_df_test,
            filtered_pop_df_test,
            filtered_alt_df_test,
            filtered_dis_df_test,
        ],
        axis=0,
        ignore_index=True,
    )
    return all_tobacco_df_train, all_tobacco_df_test


def filter_for_turmeric(scientific_df, vernacular_df, alternative_df, disinfo_df):
    topic = "turmeric"

    ## scientific
    filtered_scientific_df = scientific_df.loc[
        scientific_df["tags"] == "turmeric"
    ].head(53)
    split_ratio = calc_split_ratio(filtered_scientific_df)
    print("split_ratio turmeric", split_ratio)
    filtered_sci_df_train, filtered_sci_df_test = split_train_test_data(
        filtered_scientific_df, split_ratio
    )

    ## popular
    number_WebMD = 31
    number_HHP = 3
    number_MH = 10
    number_WH = 5
    number_MP = 0
    number_Mayo = 4
    filtered_pop_df = filter_vernacular(
        vernacular_df,
        topic,
        number_WebMD,
        number_HHP,
        number_MH,
        number_WH,
        number_MP,
        number_Mayo,
    )
    filtered_pop_df_train, filtered_pop_df_test = split_train_test_data(
        filtered_pop_df, split_ratio
    )

    ## alternative science
    number_JEBIM = 11
    number_CMT = 39
    number_homeoj = 1
    number_goethe = 0
    number_ijrh = 2
    filtered_alt_df = filter_alternative(
        alternative_df,
        topic,
        number_JEBIM,
        number_CMT,
        number_goethe,
        number_homeoj,
        number_ijrh,
    )
    filtered_alt_df_train, filtered_alt_df_test = split_train_test_data(
        filtered_alt_df, split_ratio
    )

    ## disinformation
    number_NN = 28
    number_HIN = 2
    number_MCL = 17
    number_HN = 6
    number_IW = 0
    filtered_dis_df = filter_disinformation(
        disinfo_df, topic, number_NN, number_HIN, number_MCL, number_HN, number_IW
    )
    filtered_dis_df_train, filtered_dis_df_test = split_train_test_data(
        filtered_dis_df, split_ratio
    )

    all_turmeric_df_train = pd.concat(
        [
            filtered_sci_df_train,
            filtered_pop_df_train,
            filtered_alt_df_train,
            filtered_dis_df_train,
        ],
        axis=0,
        ignore_index=True,
    )
    all_turmeric_df_test = pd.concat(
        [
            filtered_sci_df_test,
            filtered_pop_df_train,
            filtered_alt_df_test,
            filtered_dis_df_test,
        ],
        axis=0,
        ignore_index=True,
    )
    return all_turmeric_df_train, all_turmeric_df_test


def filter_for_measles(scientific_df, vernacular_df, alternative_df, disinfo_df):
    topic = "measles"

    ## scientific
    filtered_scientific_df = scientific_df.loc[scientific_df["tags"] == "measles"].head(
        22
    )
    split_ratio = calc_split_ratio(filtered_scientific_df)
    print("split_ratio measles", split_ratio)
    filtered_sci_df_train, filtered_sci_df_test = split_train_test_data(
        filtered_scientific_df, split_ratio
    )

    ## popular
    number_WebMD = 5
    number_HHP = 5
    number_MH = 3
    number_WH = 1
    number_MP = 10
    number_Mayo = 0
    filtered_pop_df = filter_vernacular(
        vernacular_df,
        topic,
        number_WebMD,
        number_HHP,
        number_MH,
        number_WH,
        number_MP,
        number_Mayo,
    )
    filtered_pop_df_train, filtered_pop_df_test = split_train_test_data(
        filtered_pop_df, split_ratio
    )

    ## alternative science
    number_JEBIM = 8
    number_CMT = 6
    number_homeoj = 8
    number_goethe = 2
    number_ijrh = 1
    filtered_alt_df = filter_alternative(
        alternative_df,
        topic,
        number_JEBIM,
        number_CMT,
        number_goethe,
        number_homeoj,
        number_ijrh,
    )
    filtered_alt_df_train, filtered_alt_df_test = split_train_test_data(
        filtered_alt_df, split_ratio
    )

    ## disinformation
    number_NN = 10
    number_HIN = 6
    number_MCL = 6
    number_HN = 0
    number_IW = 0
    filtered_dis_df = filter_disinformation(
        disinfo_df, topic, number_NN, number_HIN, number_MCL, number_HN, number_IW
    )
    filtered_dis_df_train, filtered_dis_df_test = split_train_test_data(
        filtered_dis_df, split_ratio
    )

    all_measles_df_train = pd.concat(
        [
            filtered_sci_df_train,
            filtered_pop_df_train,
            filtered_alt_df_train,
            filtered_dis_df_train,
        ],
        axis=0,
        ignore_index=True,
    )
    all_measles_df_test = pd.concat(
        [
            filtered_sci_df_test,
            filtered_pop_df_test,
            filtered_alt_df_test,
            filtered_dis_df_test,
        ],
        axis=0,
        ignore_index=True,
    )
    return all_measles_df_train, all_measles_df_test


def filter_for_inflammation(scientific_df, vernacular_df, alternative_df, disinfo_df):
    topic = "inflammation"

    ## scientific
    filtered_scientific_df = scientific_df.loc[
        scientific_df["tags"] == "inflammation"
    ].head(80)
    split_ratio = calc_split_ratio(filtered_scientific_df)
    print("split_ratio inflammation", split_ratio)
    filtered_sci_df_train, filtered_sci_df_test = split_train_test_data(
        filtered_scientific_df, split_ratio
    )

    ## popular
    number_WebMD = 16
    number_HHP = 16
    number_MH = 16
    number_WH = 16
    number_MP = 16
    number_Mayo = 0
    filtered_pop_df = filter_vernacular(
        vernacular_df,
        topic,
        number_WebMD,
        number_HHP,
        number_MH,
        number_WH,
        number_MP,
        number_Mayo,
    )
    filtered_pop_df_train, filtered_pop_df_test = split_train_test_data(
        filtered_pop_df, split_ratio
    )

    ## alternative science
    number_JEBIM = 21
    number_CMT = 21
    number_homeoj = 22
    number_goethe = 5
    number_ijrh = 11
    filtered_alt_df = filter_alternative(
        alternative_df,
        topic,
        number_JEBIM,
        number_CMT,
        number_goethe,
        number_homeoj,
        number_ijrh,
    )
    filtered_alt_df_train, filtered_alt_df_test = split_train_test_data(
        filtered_alt_df, split_ratio
    )

    ## disinformation
    number_NN = 8
    number_HIN = 21
    number_MCL = 140
    number_HN = 0
    number_IW = 1
    filtered_dis_df = filter_disinformation(
        disinfo_df, topic, number_NN, number_HIN, number_MCL, number_HN, number_IW
    )
    filtered_dis_df_train, filtered_dis_df_test = split_train_test_data(
        filtered_dis_df, split_ratio
    )

    all_inflammation_df_train = pd.concat(
        [
            filtered_sci_df_train,
            filtered_pop_df_train,
            filtered_alt_df_train,
            filtered_dis_df_train,
        ],
        axis=0,
        ignore_index=True,
    )
    all_inflammation_df_test = pd.concat(
        [
            filtered_sci_df_test,
            filtered_pop_df_test,
            filtered_alt_df_test,
            filtered_dis_df_test,
        ],
        axis=0,
        ignore_index=True,
    )
    return all_inflammation_df_train, all_inflammation_df_test


def filter_for_vaccination(scientific_df, vernacular_df, alternative_df, disinfo_df):
    topic = "vaccination"

    ## scientific
    filtered_scientific_df = scientific_df.loc[
        scientific_df["tags"] == "vaccination"
    ].head(45)
    split_ratio = calc_split_ratio(filtered_scientific_df)
    print("split_ratio vaccination", split_ratio)
    filtered_sci_df_train, filtered_sci_df_test = split_train_test_data(
        filtered_scientific_df, split_ratio
    )

    ## popular
    number_WebMD = 9
    number_HHP = 9
    number_MH = 9
    number_WH = 9
    number_MP = 9
    number_Mayo = 0
    filtered_pop_df = filter_vernacular(
        vernacular_df,
        topic,
        number_WebMD,
        number_HHP,
        number_MH,
        number_WH,
        number_MP,
        number_Mayo,
    )
    filtered_pop_df_train, filtered_pop_df_test = split_train_test_data(
        filtered_pop_df, split_ratio
    )

    ## alternative science
    number_JEBIM = 17
    number_CMT = 15
    number_homeoj = 13
    number_goethe = 2
    number_ijrh = 14
    filtered_alt_df = filter_alternative(
        alternative_df,
        topic,
        number_JEBIM,
        number_CMT,
        number_goethe,
        number_homeoj,
        number_ijrh,
    )
    filtered_alt_df_train, filtered_alt_df_test = split_train_test_data(
        filtered_alt_df, split_ratio
    )

    ## disinformation
    number_NN = 9
    number_HIN = 9
    number_MCL = 9
    number_HN = 0
    number_IW = 9
    filtered_dis_df = filter_disinformation(
        disinfo_df, topic, number_NN, number_HIN, number_MCL, number_HN, number_IW
    )
    filtered_dis_df_train, filtered_dis_df_test = split_train_test_data(
        filtered_dis_df, split_ratio
    )

    all_vaccination_df_train = pd.concat(
        [
            filtered_sci_df_train,
            filtered_pop_df_train,
            filtered_alt_df_train,
            filtered_dis_df_train,
        ],
        axis=0,
        ignore_index=True,
    )
    all_vaccination_df_test = pd.concat(
        [
            filtered_sci_df_test,
            filtered_pop_df_test,
            filtered_alt_df_test,
            filtered_dis_df_test,
        ],
        axis=0,
        ignore_index=True,
    )
    return all_vaccination_df_train, all_vaccination_df_test


def filter_for_transgender(scientific_df, vernacular_df, alternative_df, disinfo_df):
    topic = "transgender"

    ## scientific
    filtered_scientific_df = scientific_df.loc[
        scientific_df["tags"] == "transgender"
    ].head(1)
    split_ratio = calc_split_ratio(filtered_scientific_df)
    print("split_ratio transgender", split_ratio)
    filtered_sci_df_train, filtered_sci_df_test = split_train_test_data(
        filtered_scientific_df, split_ratio
    )

    ## popular
    number_WebMD = 0
    number_HHP = 1
    number_MH = 0
    number_WH = 0
    number_MP = 0
    number_Mayo = 0
    filtered_pop_df = filter_vernacular(
        vernacular_df,
        topic,
        number_WebMD,
        number_HHP,
        number_MH,
        number_WH,
        number_MP,
        number_Mayo,
    )
    filtered_pop_df_train, filtered_pop_df_test = split_train_test_data(
        filtered_pop_df, split_ratio
    )

    ## alternative science
    number_JEBIM = 1
    number_CMT = 0
    number_homeoj = 0
    number_goethe = 0
    number_ijrh = 0
    filtered_alt_df = filter_alternative(
        alternative_df,
        topic,
        number_JEBIM,
        number_CMT,
        number_goethe,
        number_homeoj,
        number_ijrh,
    )
    filtered_alt_df_train, filtered_alt_df_test = split_train_test_data(
        filtered_alt_df, split_ratio
    )

    ## disinformation
    number_NN = 0
    number_HIN = 0
    number_MCL = 0
    number_HN = 0
    number_IW = 1
    filtered_dis_df = filter_disinformation(
        disinfo_df, topic, number_NN, number_HIN, number_MCL, number_HN, number_IW
    )
    filtered_dis_df_train, filtered_dis_df_test = split_train_test_data(
        filtered_dis_df, split_ratio
    )

    all_transgender_df_train = pd.concat(
        [
            filtered_sci_df_train,
            filtered_pop_df_train,
            filtered_alt_df_train,
            filtered_dis_df_train,
        ],
        axis=0,
        ignore_index=True,
    )
    all_transgender_df_test = pd.concat(
        [
            filtered_sci_df_test,
            filtered_pop_df_test,
            filtered_alt_df_test,
            filtered_dis_df_test,
        ],
        axis=0,
        ignore_index=True,
    )
    return all_transgender_df_train, all_transgender_df_test


def filter_for_abortion(scientific_df, vernacular_df, alternative_df, disinfo_df):
    topic = "abortion"

    ## scientific
    filtered_scientific_df = scientific_df.loc[
        scientific_df["tags"] == "abortion"
    ].head(40)
    split_ratio = calc_split_ratio(filtered_scientific_df)
    print("split_ratio dementia", split_ratio)
    filtered_sci_df_train, filtered_sci_df_test = split_train_test_data(
        filtered_scientific_df, split_ratio
    )

    ## popular
    number_WebMD = 14
    number_HHP = 3
    number_MH = 9
    number_WH = 14
    number_MP = 0
    number_Mayo = 0
    filtered_pop_df = filter_vernacular(
        vernacular_df,
        topic,
        number_WebMD,
        number_HHP,
        number_MH,
        number_WH,
        number_MP,
        number_Mayo,
    )
    filtered_pop_df_train, filtered_pop_df_test = split_train_test_data(
        filtered_pop_df, split_ratio
    )

    ## alternative science
    number_JEBIM = 12
    number_CMT = 13
    number_homeoj = 15
    number_goethe = 1
    number_ijrh = 3
    filtered_alt_df = filter_alternative(
        alternative_df,
        topic,
        number_JEBIM,
        number_CMT,
        number_goethe,
        number_homeoj,
        number_ijrh,
    )
    filtered_alt_df_train, filtered_alt_df_test = split_train_test_data(
        filtered_alt_df, split_ratio
    )

    ## disinformation
    number_NN = 29
    number_HIN = 18
    number_MCL = 1
    number_HN = 0
    number_IW = 2
    filtered_dis_df = filter_disinformation(
        disinfo_df, topic, number_NN, number_HIN, number_MCL, number_HN, number_IW
    )
    filtered_dis_df_train, filtered_dis_df_test = split_train_test_data(
        filtered_dis_df, split_ratio
    )

    all_abortion_df_train = pd.concat(
        [
            filtered_sci_df_train,
            filtered_pop_df_train,
            filtered_alt_df_train,
            filtered_dis_df_train,
        ],
        axis=0,
        ignore_index=True,
    )
    all_abortion_df_test = pd.concat(
        [
            filtered_sci_df_test,
            filtered_pop_df_test,
            filtered_alt_df_test,
            filtered_dis_df_test,
        ],
        axis=0,
        ignore_index=True,
    )
    return all_abortion_df_train, all_abortion_df_test


def filter_for_climatechange(scientific_df, vernacular_df, alternative_df, disinfo_df):
    topic = "climatechange"

    ## scientific
    # filtered_scientific_df = scientific_df.loc[scientific_df['tags']=='dementia']
    filtered_sci_df = scientific_df[
        scientific_df["tags"].str.contains("climatechange", case=False, na=False)
    ].head(15)
    split_ratio = calc_split_ratio(filtered_sci_df)
    print("split_ratio climatechange", split_ratio)
    filtered_sci_df_train, filtered_sci_df_test = split_train_test_data(
        filtered_sci_df, split_ratio
    )

    ## popular
    number_WebMD = 3
    number_HHP = 3
    number_MH = 3
    number_WH = 3
    number_MP = 3
    number_Mayo = 0
    filtered_pop_df = filter_vernacular(
        vernacular_df,
        topic,
        number_WebMD,
        number_HHP,
        number_MH,
        number_WH,
        number_MP,
        number_Mayo,
    )
    filtered_pop_df_train, filtered_pop_df_test = split_train_test_data(
        filtered_pop_df, split_ratio
    )

    ## alternative science
    number_JEBIM = 1
    number_CMT = 15
    number_goethe = 0
    number_homeoj = 0
    number_ijrh = 0
    filtered_alt_df = filter_alternative(
        alternative_df,
        topic,
        number_JEBIM,
        number_CMT,
        number_goethe,
        number_homeoj,
        number_ijrh,
    )
    filtered_alt_df_train, filtered_alt_df_test = split_train_test_data(
        filtered_alt_df, split_ratio
    )

    ## disinformation
    number_NN = 7
    number_HIN = 7
    number_MCL = 1
    number_HN = 0
    number_IW = 0
    filtered_dis_df = filter_disinformation(
        disinfo_df, topic, number_NN, number_HIN, number_MCL, number_HN, number_IW
    )
    filtered_dis_df_train, filtered_dis_df_test = split_train_test_data(
        filtered_dis_df, split_ratio
    )

    all_climatechange_train = pd.concat(
        [
            filtered_sci_df_train,
            filtered_pop_df_train,
            filtered_alt_df_train,
            filtered_dis_df_train,
        ],
        axis=0,
        ignore_index=True,
    )
    all_climatechange_test = pd.concat(
        [
            filtered_sci_df_test,
            filtered_pop_df_test,
            filtered_alt_df_test,
            filtered_dis_df_test,
        ],
        axis=0,
        ignore_index=True,
    )
    return all_climatechange_train, all_climatechange_test


def filter_for_pandemics(scientific_df, vernacular_df, alternative_df, disinfo_df):
    topic = "pandemics"

    ## scientific
    filtered_sci_df = scientific_df[
        scientific_df["tags"].str.contains("pandemics", case=False, na=False)
    ].head(60)
    split_ratio = calc_split_ratio(filtered_sci_df)
    print("split_ratio pandemics", split_ratio)
    filtered_sci_df_train, filtered_sci_df_test = split_train_test_data(
        filtered_sci_df, split_ratio
    )

    ## popular
    number_WebMD = 12
    number_HHP = 12
    number_MH = 12
    number_WH = 12
    number_MP = 12
    number_Mayo = 0
    filtered_pop_df = filter_vernacular(
        vernacular_df,
        topic,
        number_WebMD,
        number_HHP,
        number_MH,
        number_WH,
        number_MP,
        number_Mayo,
    )
    filtered_pop_df_train, filtered_pop_df_test = split_train_test_data(
        filtered_pop_df, split_ratio
    )

    ## alternative science
    number_JEBIM = 3
    number_CMT = 56
    number_goethe = 0
    number_homeoj = 7
    number_ijrh = 4
    filtered_alt_df = filter_alternative(
        alternative_df,
        topic,
        number_JEBIM,
        number_CMT,
        number_goethe,
        number_homeoj,
        number_ijrh,
    )
    filtered_alt_df_train, filtered_alt_df_test = split_train_test_data(
        filtered_alt_df, split_ratio
    )

    ## disinformation
    number_NN = 25
    number_HIN = 20
    number_MCL = 3
    number_HN = 5
    number_IW = 7
    filtered_dis_df = filter_disinformation(
        disinfo_df, topic, number_NN, number_HIN, number_MCL, number_HN, number_IW
    )
    filtered_dis_df_train, filtered_dis_df_test = split_train_test_data(
        filtered_dis_df, split_ratio
    )

    all_pandemics_train = pd.concat(
        [
            filtered_sci_df_train,
            filtered_pop_df_train,
            filtered_alt_df_train,
            filtered_dis_df_train,
        ],
        axis=0,
        ignore_index=True,
    )
    all_pandemics_test = pd.concat(
        [
            filtered_sci_df_test,
            filtered_pop_df_test,
            filtered_alt_df_test,
            filtered_dis_df_test,
        ],
        axis=0,
        ignore_index=True,
    )
    return all_pandemics_train, all_pandemics_test


def filter_for_urine(scientific_df, vernacular_df, alternative_df, disinfo_df):
    topic = "urine"

    ## scientific
    # filtered_scientific_df = scientific_df.loc[scientific_df['tags']=='dementia']
    filtered_sci_df = scientific_df[
        scientific_df["tags"].str.contains("urine", case=False, na=False)
    ].head(55)
    split_ratio = calc_split_ratio(filtered_sci_df)
    print("split_ratio urine", split_ratio)
    filtered_sci_df_train, filtered_sci_df_test = split_train_test_data(
        filtered_sci_df, split_ratio
    )
    ## popular
    number_WebMD = 12
    number_HHP = 12
    number_MH = 12
    number_WH = 12
    number_MP = 7
    number_Mayo = 0
    filter_vern_df = filter_vernacular(
        vernacular_df,
        topic,
        number_WebMD,
        number_HHP,
        number_MH,
        number_WH,
        number_MP,
        number_Mayo,
    )
    filter_vern_df_train, filter_vern_df_test = split_train_test_data(
        filter_vern_df, split_ratio
    )

    ## alternative science
    number_JEBIM = 19
    number_CMT = 20
    number_goethe = 0
    number_homeoj = 10
    number_ijrh = 6
    filtered_alt_df = filter_alternative(
        alternative_df,
        topic,
        number_JEBIM,
        number_CMT,
        number_goethe,
        number_homeoj,
        number_ijrh,
    )
    filtered_alt_df_train, filtered_alt_df_test = split_train_test_data(
        filtered_alt_df, split_ratio
    )

    ## disinformation
    number_NN = 23
    number_HIN = 13
    number_MCL = 16
    number_HN = 2
    number_IW = 1
    filtered_dis_df = filter_disinformation(
        disinfo_df, topic, number_NN, number_HIN, number_MCL, number_HN, number_IW
    )
    filtered_dis_df_train, filtered_dis_df_test = split_train_test_data(
        filtered_dis_df, split_ratio
    )

    all_urine_train = pd.concat(
        [
            filtered_sci_df_train,
            filter_vern_df_train,
            filtered_alt_df_train,
            filtered_dis_df_train,
        ],
        axis=0,
        ignore_index=True,
    )
    all_urine_test = pd.concat(
        [
            filtered_sci_df_test,
            filter_vern_df_test,
            filtered_alt_df_test,
            filtered_dis_df_test,
        ],
        axis=0,
        ignore_index=True,
    )

    return all_urine_train, all_urine_test


def calc_split_ratio(dataset):
    # 85% training (70% train,+ 15% val) 15% test
    split_ratio = int(len(dataset) * 0.80)
    return split_ratio


def split_train_test_data(dataset, split_ratio):
    # Split the data into training and validation sets
    train_dataset, val_dataset = np.split(dataset, [split_ratio])
    return train_dataset, val_dataset


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("base_url")  # Path to the base_url containing input files also for output files
    args = argparser.parse_args()

    base_url = args.base_url
    col_list = ["category_id", "tags", "data-source", "text"]

    scientific_df_path = load_dataset_sci(base_url, col_list)
    vernacular_df_path = load_dataset_vern(base_url, col_list)
    alternative_df_path = load_dataset_alt(base_url, col_list)
    disinfo_df_path  = load_dataset_dis(base_url, col_list)
    print("datasets loaded")

    scientific_df = pd.read_csv(scientific_df_path)
    vernacular_df = pd.read_csv(vernacular_df_path)
    alternative_df = pd.read_csv(alternative_df_path)
    disinfo_df = pd.read_csv(disinfo_df_path)

    ## filter for topics
    all_cumin_train, all_cumin_test = filter_for_cumin(
        scientific_df, vernacular_df, alternative_df, disinfo_df
    )
    
    all_dementia_train, all_dementia_test = filter_for_dementia(
        scientific_df, vernacular_df, alternative_df, disinfo_df
    )
    all_heartattack_train, all_heartattack_test = filter_for_heartattack(
        scientific_df, vernacular_df, alternative_df, disinfo_df
    )
    all_insomnia_train, all_insomnia_test = filter_for_insomnia(
        scientific_df, vernacular_df, alternative_df, disinfo_df
    )
    all_menopause_train, all_menopause_test = filter_for_menopause(
        scientific_df, vernacular_df, alternative_df, disinfo_df
    )
    all_stroke_train, all_stroke_test = filter_for_stroke(
        scientific_df, vernacular_df, alternative_df, disinfo_df
    )
    all_tobacco_train, all_tobacco_test = filter_for_tobacco(
        scientific_df, vernacular_df, alternative_df, disinfo_df
    )
    all_turmeric_train, all_turmeric_test = filter_for_turmeric(
        scientific_df, vernacular_df, alternative_df, disinfo_df
    )
    all_measles_train, all_measles_test = filter_for_measles(
        scientific_df, vernacular_df, alternative_df, disinfo_df
    )
    all_inflammation_train, all_inflammation_test = filter_for_inflammation(
        scientific_df, vernacular_df, alternative_df, disinfo_df
    )
    all_vaccination_train, all_vaccination_test = filter_for_vaccination(
        scientific_df, vernacular_df, alternative_df, disinfo_df
    )
    # all_transgender_train,all_transgender_test = filter_for_transgender(scientific_df, vernacular_df, alternative_df, disinfo_df)

    all_abortion_train, all_abortion_test = filter_for_abortion(
        scientific_df, vernacular_df, alternative_df, disinfo_df
    )
    all_climatechange_train, all_climatechange_test = filter_for_climatechange(
        scientific_df, vernacular_df, alternative_df, disinfo_df
    )
    all_pandemics_train, all_pandemics_test = filter_for_pandemics(
        scientific_df, vernacular_df, alternative_df, disinfo_df
    )
    all_urine_train, all_urine_test = filter_for_urine(
        scientific_df, vernacular_df, alternative_df, disinfo_df
    )

    print("cumin", all_cumin_train.shape)
    print("dementia", all_dementia_train.shape)
    print("heartattack", all_heartattack_train.shape)
    print("insomnia", all_insomnia_train.shape)
    print("menopause", all_menopause_train.shape)
    print("stroke", all_stroke_train.shape)
    print("tobacco", all_tobacco_train.shape)
    print("turmeric", all_turmeric_train.shape)
    print("measles", all_measles_train.shape)
    print("inflammation", all_inflammation_train.shape)
    print("vaccination", all_vaccination_train.shape)
    # print('transgender', all_transgender_train.shape)
    print("abortion", all_abortion_train.shape)
    print("climate change", all_climatechange_train.shape)
    print("pandemics", all_pandemics_train.shape)
    print("urine", all_urine_train.shape)

    print("filtered for topics and data sources")
    all_df_list_train = [
        all_cumin_train,
        all_dementia_train,
        all_heartattack_train,
        all_insomnia_train,
        all_menopause_train,
        all_stroke_train,
        all_tobacco_train,
        all_turmeric_train,
        all_measles_train,
        all_inflammation_train,
        all_vaccination_train,
        all_abortion_train,
        all_climatechange_train,
        all_pandemics_train,
        all_urine_train,
    ]
    all_df_list_test = [
        all_cumin_test,
        all_dementia_test,
        all_heartattack_test,
        all_insomnia_test,
        all_menopause_test,
        all_stroke_test,
        all_tobacco_test,
        all_turmeric_test,
        all_measles_test,
        all_inflammation_test,
        all_vaccination_test,
        all_abortion_test,
        all_climatechange_test,
        all_pandemics_test,
        all_urine_test,
    ]
    all_df_train = pd.concat(all_df_list_train, axis=0, ignore_index=True)
    all_df_test = pd.concat(all_df_list_test, axis=0, ignore_index=True)

    complete_df = pd.concat([all_df_train, all_df_test], axis=0, ignore_index=True)
    complete_df["text"] = complete_df["text"].str.strip()

    print(all_df_train.shape)
    print(all_df_test.shape)
    all_df_train.to_csv(
        base_url + "FSoLS_TRAIN-VAL.csv",
        index=False,
    )
    all_df_test.to_csv(
        base_url + "FSoLS_TEST.csv", index=False
    )

    complete_df.to_csv(
        base_url + "FSoLS_COMPLETE.csv",
        index=False,
    )
    print("done")


if __name__ == "__main__":
    main()

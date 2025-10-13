from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from folktables import (
    ACSDataSource,
    adult_filter,
    travel_time_filter,
    BasicProblem,
    public_coverage_filter,
)
import pandas as pd
import numpy as np


def get_adult_sex():
    return get_adult_acs(sensitive_attribute="SEX")


def get_adult_race():
    return get_adult_acs(sensitive_attribute="RAC1P")


def get_adult_acs(states=["AL"], sensitive_attribute="SEX"):
    features = [
        "AGEP",
        "COW",
        "SCHL",
        "MAR",
        "OCCP",
        "POBP",
        "RELP",
        "WKHP",
        "SEX",
        "RAC1P",
    ]
    features = [f for f in features if f != sensitive_attribute]
    ACSAdult = BasicProblem(
        features=features,
        target="PINCP",
        target_transform=lambda x: x > 50000,
        group=sensitive_attribute,
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    ca_data = data_source.get_data(states=states, download=True)
    X, y, s = ACSAdult.df_to_numpy(ca_data)
    s[s == 2] = 0
    y = np.squeeze(y)
    s = np.squeeze(s)

    X_d1, X_d2, y_d1, y_d2, s_d1, s_d2 = train_test_split(
        X, y, s, test_size=0.2, random_state=1, stratify=y
    )

    scaler1 = StandardScaler()
    X_d1 = scaler1.fit_transform(X_d1)
    scaler2 = StandardScaler()
    X_d2 = scaler2.fit_transform(X_d2)

    return (X_d1, y_d1, s_d1), (X_d2, y_d2, s_d2)


def get_acs_employment_race(sensitive_attribute="RAC1P"):
    return get_acs_employment(sensitive_attribute=sensitive_attribute)


def get_acs_employment(states=["AL"], sensitive_attribute="SEX"):
    features = [
        "AGEP",
        "SCHL",
        "MAR",
        "RELP",
        "DIS",
        "ESP",
        "CIT",
        "MIG",
        "MIL",
        "ANC",
        "NATIVITY",
        "DEAR",
        "DEYE",
        "DREM",
        "SEX",
        "RAC1P",
    ]
    features = [f for f in features if f != sensitive_attribute]
    ACSEmployment = BasicProblem(
        features=features,
        target="ESR",
        target_transform=lambda x: x == 1,
        group=sensitive_attribute,
        preprocess=lambda x: x,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    ca_data = data_source.get_data(states=states, download=True)
    X, y, s = ACSEmployment.df_to_numpy(ca_data)
    s[s == 2] = 0
    y = np.squeeze(y)
    s = np.squeeze(s)

    X_d1, X_d2, y_d1, y_d2, s_d1, s_d2 = train_test_split(
        X, y, s, test_size=0.2, random_state=1, stratify=y
    )

    scaler1 = StandardScaler()
    X_d1 = scaler1.fit_transform(X_d1)
    scaler2 = StandardScaler()
    X_d2 = scaler2.fit_transform(X_d2)

    return (X_d1, y_d1, s_d1), (X_d2, y_d2, s_d2)


def get_travel(states=["AL"], sensitive_attribute="SEX"):
    features = [
        "AGEP",
        "SCHL",
        "MAR",
        "SEX",
        "DIS",
        "ESP",
        "MIG",
        "RELP",
        "RAC1P",
        "PUMA",
        "ST",
        "CIT",
        "OCCP",
        "JWTR",
        "POWPUMA",
        "POVPIP",
    ]
    features = [f for f in features if f != sensitive_attribute]
    ACSTravelTime = BasicProblem(
        features=features,
        target="JWMNP",
        target_transform=lambda x: x > 20,
        group=sensitive_attribute,
        preprocess=travel_time_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=states, download=True)
    features, label, group = ACSTravelTime.df_to_numpy(acs_data)

    group[group == 2] = 0
    label = label * 1

    X_d1, X_d2, y_d1, y_d2, s_d1, s_d2 = train_test_split(
        features, label, group, test_size=0.2, random_state=1
    )

    scaler1 = StandardScaler()
    X_d1 = scaler1.fit_transform(X_d1)
    scaler2 = StandardScaler()
    X_d2 = scaler2.fit_transform(X_d2)

    return (X_d1, y_d1, s_d1), (X_d2, y_d2, s_d2)


def get_mobility(states=["AL"], sensitive_attribute="SEX"):
    features = [
        "AGEP",
        "SCHL",
        "MAR",
        "SEX",
        "DIS",
        "ESP",
        "CIT",
        "MIL",
        "ANC",
        "NATIVITY",
        "RELP",
        "DEAR",
        "DEYE",
        "DREM",
        "RAC1P",
        "GCL",
        "COW",
        "ESR",
        "WKHP",
        "JWMNP",
        "PINCP",
    ]
    features = [f for f in features if f != sensitive_attribute]
    ACSMobility = BasicProblem(
        features=features,
        target="MIG",
        target_transform=lambda x: x == 1,
        group=sensitive_attribute,
        preprocess=lambda x: x.drop(x.loc[(x["AGEP"] <= 18) | (x["AGEP"] >= 35)].index),
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=states, download=True)
    features, label, group = ACSMobility.df_to_numpy(acs_data)

    group[group == 2] = 0
    label = label * 1

    X_d1, X_d2, y_d1, y_d2, s_d1, s_d2 = train_test_split(
        features, label, group, test_size=0.2, random_state=1
    )

    scaler1 = StandardScaler()
    X_d1 = scaler1.fit_transform(X_d1)
    scaler2 = StandardScaler()
    X_d2 = scaler2.fit_transform(X_d2)

    return (X_d1, y_d1, s_d1), (X_d2, y_d2, s_d2)


def get_public_coverage_sex():
    return get_public_coverage(sensitive_attribute="SEX")


def get_public_coverage(states=["AL"], sensitive_attribute="SEX"):
    features = [
        "AGEP",
        "SCHL",
        "MAR",
        "SEX",
        "DIS",
        "ESP",
        "CIT",
        "MIG",
        "MIL",
        "ANC",
        "NATIVITY",
        "DEAR",
        "DEYE",
        "DREM",
        "PINCP",
        "ESR",
        "ST",
        "FER",
        "RAC1P",
    ]
    features = [f for f in features if f != sensitive_attribute]
    ACSPublicCoverage = BasicProblem(
        features=features,
        target="PUBCOV",
        target_transform=lambda x: x == 1,
        group=sensitive_attribute,
        preprocess=public_coverage_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=states, download=True)
    features, label, group = ACSPublicCoverage.df_to_numpy(acs_data)

    group[group == 2] = 0
    label = label * 1

    X_d1, X_d2, y_d1, y_d2, s_d1, s_d2 = train_test_split(
        features, label, group, test_size=0.2,random_state=1
    )

    scaler1 = StandardScaler()
    X_d1 = scaler1.fit_transform(X_d1)
    scaler2 = StandardScaler()
    X_d2 = scaler2.fit_transform(X_d2)

    return (X_d1, y_d1, s_d1), (X_d2, y_d2, s_d2)


def get_german_age(sensitive_attrib="age_geq_median", base_dir="./"):
    data = pd.read_csv("{}preprocessing/processed_german.csv".format(base_dir))
    # data = data.dropna(axis=0)
    X = data.drop(
        ["target", sensitive_attrib],
        axis=1,
    ).values
    y = data["target"].values
    s = data[sensitive_attrib].values

    X_d1, X_d2, y_d1, y_d2, s_d1, s_d2 = train_test_split(X, y, s, test_size=0.20)

    return (X_d1, y_d1, s_d1), (X_d2, y_d2, s_d2)


def get_diabetes_race(sensitive_attrib="race", base_dir="./"):
    data = pd.read_csv(
        "{}preprocessing/processed_diabetes_readmission.csv".format(base_dir)
    )
    # data = data.dropna(axis=0)
    # Save DataFrame
    X = data.drop(
        ["target", sensitive_attrib],
        axis=1,
    ).values
    y = data["target"].values
    s = data[sensitive_attrib].values

    X_d1, X_d2, y_d1, y_d2, s_d1, s_d2 = train_test_split(X, y, s, test_size=0.20, random_state=1)

    return (X_d1, y_d1, s_d1), (X_d2, y_d2, s_d2)


def get_celeba_attract(sensitive_attrib="Male", base_dir="./", target="Attractive"):
    return get_celeba(
        target=target, sensitive_attrib=sensitive_attrib, base_dir=base_dir
    )


def get_celeba(sensitive_attrib="Male", base_dir="./", target="Smiling"):
    data1 = pd.read_csv(
        "{}preprocessing/celeba.data1.csv".format(base_dir), index_col=False
    )

    data2 = pd.read_csv(
        "{}preprocessing/celeba.data2.csv".format(base_dir), index_col=False
    )

    def process(df):
        X = df.loc[:, df.columns != target]
        y = df[target].values

        s = X[sensitive_attrib].values
        X = X.drop(sensitive_attrib, axis=1)

        # norms = np.linalg.norm(X, axis=1)
        # X[norms > 2] *= 6 / norms[norms > 6][:, None]
        s[s == -1] = 0
        y[y == -1] = 0
        return X.values, y, s

    return process(data1), process(data2)


def get_old_adult(include_y_in_x=False, base_dir="./"):
    #
    def get_data(df, include_y_in_x=include_y_in_x):
        if "gender_ Male" in df.columns:
            S = df["gender_ Male"].values
            X = df.drop("gender_ Male", axis=1)
        if not include_y_in_x:
            X = df.drop(["outcome_ >50K", "gender_ Male"], axis=1).values
        else:
            X = df.values

        y = df["outcome_ >50K"].values

        return X, y, S

    data1 = pd.read_csv(
        "{}preprocessing/adult.data1.csv".format(base_dir), index_col=False
    )
    data2 = pd.read_csv(
        "{}preprocessing/adult.data2.csv".format(base_dir), index_col=False
    )
    data1 = data1.sample(frac=1)
    return get_data(data1), get_data(data2)

def get_dataset_registry():
    datasets = {
        "adult_income": get_adult_acs,
        "celeba_attract": get_celeba_attract,
        "employment": get_acs_employment,
        "mobility": get_mobility,
        "coverage": get_public_coverage,
        "travel": get_travel,
        "german_age": get_german_age,
        "diabetes_race": get_diabetes_race,
        "old_adult": get_old_adult,
    }

    return datasets


if __name__ == "__main__":
    data1, data2 = get_dataset_registry()["old_adult"]()
    X_d1, y_d1, s_d1 = data1
    X_d2, y_d2, s_d2 = data2
    X = np.concatenate((X_d1, X_d2), axis=0)
    y = np.concatenate((y_d1, y_d2), axis=0)
    s = np.concatenate((s_d1, s_d2), axis=0)
    print(X.shape, y.shape, s.shape)
    count_0 = (s == 0).sum()/len(s)
    count_1 = (s == 1).sum()/len(s)
    print(f"Count of 0: {count_0}, Count of 1: {count_1}") 
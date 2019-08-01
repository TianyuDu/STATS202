import pandas as pd
import numpy as np

import sys
sys.path.append("../")
from CONSTANTS import PANSS

def load_individual_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=0)
    return df


def gen_label(df: pd.DataFrame) -> pd.DataFrame:
    df_cp = df.copy()
    # Either flagged or assigned to CS:
    df_cp["Alert"] = 1 - (df_cp["LeadStatus"] == "Passed").astype(np.int)
    return df_cp


def load_whole(path: str) -> pd.DataFrame:
    collect, lengths = [], []
    if not path.endswith("/"):
        raise ValueError("Path should be a directory (i.e. ends with /)")
    for v in ["A", "B", "C", "D"]:
        df_temp = load_individual_dataset(path + "Study_{}.csv".format(v))
        collect.append(df_temp)
        lengths.append(len(df_temp))
    df = pd.concat(collect)
    df.reset_index(inplace=True, drop=True)
    assert len(df) == sum(lengths)
    return gen_label(df)


def gen_slp_assessment(
        raw: pd.DataFrame,
        keep_patient_ID: bool = True,
        keep_assessment_ID: bool = True,
        keep_visit_day: bool = True,
) -> (pd.DataFrame, pd.DataFrame, list):
    """
    Generates supervised learning problem for the classification problem.
    Each row corresponds to an assessment.
    Args:
        raw: the raw dataset loaded from csv.
        keep_patient_ID: keep patient ID in the desgin data frame X.
        keep_assessment_ID: keep assessment ID in the design data frame X.
    Returns:
        (X, y): Dataframes of the supervised learning problem.
        FEATURE: A list containing features used for model training.
        PANSS: A list containing panss column names.
    """
    # The list of features in design data frame returned.
    SELECT = ["Country", "TxGroup", "PANSS_Total"]

    SELECT.extend(PANSS)
    if keep_patient_ID:
        SELECT.append("PatientID")
    if keep_assessment_ID:
        SELECT.append("AssessmentiD")
    if keep_visit_day:
        SELECT.append("VisitDay")
    # Create dummy variables for countries
    X = create_dummies(raw[SELECT], columns=["Country"])
    # Convert Treatment Indictor to 0-1 dummy
    X["Treatment"] = (X["TxGroup"] == "Treatment").astype(int)
    X.drop(columns=["TxGroup"], inplace=True)
    FEATURE = list(X.columns.copy())
    if keep_patient_ID:
        FEATURE.remove("PatientID")
    if keep_assessment_ID:
        FEATURE.remove("AssessmentiD")
    if keep_visit_day:
        FEATURE.remove("VisitDay")
    y = raw["Alert"]
    return X, y, FEATURE, PANSS


def create_dummies(
        df: pd.DataFrame,
        columns: list,
) -> pd.DataFrame:
    """
    Change selected quantitive data.
    """
    X = df.copy()
    dummies = pd.get_dummies(X[columns])
    X.drop(columns=columns, inplace=True)
    X_with_dummy = pd.concat([X, dummies], axis=1)
    # Create an other-country dummy, since there are 20 UK samples in testing set
    # but there is no UK sample in training set
    X_with_dummy["Country_Other"] = int(0)
    return X_with_dummy


def parse_test_set(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Wraps the operations taken for parsing country and treatments.
    """
    parsed = parse_test_set_countries(df_train, df_test)
    return parse_test_set_treatment(parsed)[df_train.columns]


def parse_test_set_treatment(
        df_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parse the treatment dummy variable in the testing set.
    """
    parsed_test = df_test.copy()
    parsed_test["Treatment"] = (
        parsed_test["TxGroup"] == "Treatment").astype(int)
    parsed_test.drop(columns=["TxGroup"], inplace=True)
    return parsed_test


def parse_test_set_countries(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parses the country dummies.
    """
    parsed_test = df_test.copy()
    country_list = [
        col
        for col in df_train.columns
        if col.startswith("Country")
    ]
    for country in country_list:
        # Create empty dummy variables
        parsed_test[country] = int(0)
    for i in range(len(df_test)):
        country = parsed_test.at[i, "Country"]
        if "Country_{}".format(country) in country_list:
            parsed_test.at[i, "Country_{}".format(country)] = int(1)
        else:
            parsed_test.at[i, "Country_Other"] = int(1)
    parsed_test.drop(columns=["Country"], inplace=True)
    return parsed_test


if __name__ == "__main__":
    df = load_whole("./data/")

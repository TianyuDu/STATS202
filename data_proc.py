import pandas as pd
import numpy as np


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
        df_temp = load_individual_dataset(path + f"Study_{v}.csv")
        collect.append(df_temp)
        lengths.append(len(df_temp))
    df = pd.concat(collect)
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
        PANSS: A list containing panss names.
    """
    # The list of features in design data frame returned.
    SELECT = ["Country", "TxGroup"]
    # List of PANSS Scores.
    PANSS = [f"P{i}" for i in range(1, 8)] \
        + [f"N{i}" for i in range(1, 8)] \
        + [f"G{i}" for i in range(1, 17)] \
        + ["PANSS_Total"]

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


if __name__ == "__main__":
    df = load_whole("./data/")

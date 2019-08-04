"""
Jul. 28, 2019
This script contains data processing utilities for
the regression setting.

For the classification task, training instances are indexed using
AssessmentIDs, and for the regression task, trainining instances
are indexed using PatientIDs.
"""
import sys
from typing import Optional, Union
import pandas as pd
import numpy as np

from tqdm import tqdm

sys.path.append("../")
from util import data_proc
from util import features
from CONSTANTS import PANSS


def select_patient(
        df_test: pd.DataFrame,
        sample_submission: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filters the patient ids, return a sub-dataframe of df_test that
    contains patient IDs in sample submission.
    Args:
        Both df_test and sample_submission must contains "PatientID" column.
    """
    if "PatientID" not in df_test.columns:
        raise KeyError("df_test must have PatientID column.")
    if "PatientID" not in sample_submission.columns:
        raise KeyError("sample_submission must have PatientID column.")
    valid_idx = [
        (x in list(sample_submission["PatientID"]))
        for x in df_test["PatientID"]
    ]
    selected = df_test[valid_idx]
    print("Propotion selected: {:0.6f}".format(len(selected) / len(df_test)))
    return selected


def gen_slp_patient(
        df_assessment: pd.DataFrame,
        include_label: bool = True,
        min_visit: int = None,
) -> (pd.DataFrame, Optional[pd.DataFrame]):
    """
    Reshape dataset so that each row of the new dataset corresponds to one
    Patient instead of one Assessment.
    Args:
        df_assessment:
            A dataframe with rows indexed by Assessment ID.
        
        include_label:
            A bool indicating wether to generate the label (last visit PANSS total)
            from the dataset. Use True when applying on the training set, so that the
            method returns (X, y), use False on the test set, so that only X is returned.

        min_visit:
            Only consider patients with more checkpoints than min_visit.
            If the number of observations is less than min_visit for one
            patient, it might be the case that this patient dropped out eariler.
    Returns:
        df_patient:
            A dataframe with rows indexed by Patient ID.
    """
    if "PatientID" not in df_assessment:
        raise KeyError("df_assessment DataFame must contain a 'PatientID' column.")
    patient_ids = list(set(df_assessment["PatientID"]))

    def retrive_info(pid: int):
        return df_assessment[df_assessment["PatientID"] == pid]

    X_lst, y_lst = [], []
    print("Creating Patient Information...")
    for pid in tqdm(patient_ids):
        individual_info = retrive_info(pid)
        if min_visit is None or len(individual_info) >= min_visit:
            patient_X, patient_y = reduce_patient_features(
                individual_info, include_label=include_label)
            X_lst.append(patient_X)
            y_lst.append(patient_y)
        else:
            # print("Patient dropped.")
            pass

    print("Number of patients found: {}".format(len(X_lst)))
    X = pd.concat(X_lst)
    if include_label:
        y = pd.DataFrame({"Final_PANSS_Total": y_lst})
        assert len(X) == len(y)
        return X, y
    else:
        return X


def reduce_patient_features(
        patient_info: pd.DataFrame,
        include_label: bool,
) -> (pd.DataFrame, Union[pd.DataFrame, None]):
    PANSS.append("PANSS_Total")
    # Assert all assessments belong to single patient.
    if len(set(patient_info["PatientID"])) != 1:
        raise ValueError(
            "Collection of patient assessments should belong to one single patient, got: {}".format(
                set(patient_info['PatientID']))
        )
    reduced = dict()
    info = patient_info.reset_index(drop=True)
    if include_label:
        if len(info) == 1:
            raise ValueError(
                "Include label is True, but there is only one assessment for patient.")
        # Generate the label
        y = info.iloc[-1, :]["PANSS_Total"]
        # Drop the last assessment.
        info.drop([info.index[-1]], axis=0, inplace=True)
    else:
        y = None

    def add_feature(src: pd.DataFrame, target: dict, prefix: str = None) -> None:
        if prefix is None:
            prefix = ""
        else:
            prefix += "_"
        for k, v in src.items():
            target[str(prefix) + str(k)] = v

    # *** Create TxGroup Dummies ***
    reduced["Treatment"] = int(info["Treatment"][0])
    # *** Create Country Dummies ***
    country_list = [x for x in info.columns if x.startswith("Country")]
    country_one_hot = info[country_list].iloc[0, :]
    add_feature(country_one_hot, reduced, None)
    # *** Create Initial Measures ***
    add_feature(info.iloc[0, :][PANSS], reduced, "initial")
    # *** Create Last (before the 18-th week) Measures ***
    add_feature(info.iloc[-1, :][PANSS], reduced, "last")
    # *** Create Mean Measures ***
    add_feature(info[PANSS].mean(), reduced, "mean")
    # *** Create Std Measures ***
    add_feature(info[PANSS].std(), reduced, "std")
    # *** Max and Min ***
    add_feature(info[PANSS].min(), reduced, "min")
    add_feature(info[PANSS].max(), reduced, "max")
    # Convert values to lists of values so that reduced
    # is compaticable with DataFrame.
    for k, v in reduced.items():
        reduced[k] = [v]
    return pd.DataFrame.from_dict(reduced), y


def prepare_data():
    df_train = data_proc.load_whole(path="../data/")
    df_test = pd.read_csv("../data/Study_E.csv", header=0)
    # Reduced countries
    major_countries = ["USA", "Russia", "Ukraine", "China", "Japan"]
    df_train = features.reduce_countries(df_train, major_countries)
    df_test = features.reduce_countries(df_test, major_countries)
    # Create treatment column
    df_train["Treatment"] = (df_train.TxGroup == "Treatment").astype(int)
    df_train.drop(columns=["LeadStatus"], inplace=True)
    df_test["Treatment"] = (df_test.TxGroup == "Treatment").astype(int)
    # df_test.drop(columns=["TxGroup"], inplace=True)
    # Drop unnecessary features
    df_train.drop(
        columns=["Study", "SiteID", "RaterID", "VisitDay", "Alert"],
        inplace=True)
    # df_test.drop(
    #     columns=["Study", "SiteID", "RaterID", "VisitDay"],
    #     inplace=True)
    df_train = pd.get_dummies(df_train, prefix="Country")
    # df_train.drop(columns=["Country"], inplace=True)
    df_test = data_proc.parse_test_set(df_train, df_test)
    # df_test.drop(columns=["Country"], inplace=True)
    # Create dummy variables for country
    f = lambda x: x.drop(
        columns=["Country_Control", "Country_Treatment"], inplace=True)
    f(df_train)
    f(df_test)
    print("Training set shape: {}".format(df_train.shape))
    print("Test set shape: {}".format(df_test.shape))
    return df_train, df_test

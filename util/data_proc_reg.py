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

sys.path.append("../")


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
    print(f"Propotion selected: {len(selected) / len(df_test): 0.6f}")
    return selected


def convert_to_patient(
        df_assessment: pd.DataFrame,
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
    if not "PatientID" in df_assessment:
        raise KeyError("df_assessment DataFame must contain a 'PatientID' column.")
    patient_ids = list(set(df_assessment["PatientID"]))

    def retrive_info(pid: int):
        return df_assessment[df_assessment["PatientID"] == pid]

    X_lst, y_lst = [], []
    for pid in patient_ids:
        patient_X, patient_y = reduce_patient_features(retrive_info(pid), include_label=include_label)
        if min_visit is None or len(patient_X) >= min_visit:
            X_lst.append(patient_X)
            y_lst.append(patient_y)
        else:
            print("Patient dropped.")

    print(f"Numer of patients found: {len(X_lst)}")
    X = pd.concat(X_lst)
    if include_label:
        y = pd.concat(y_lst)
        assert len(X) == len(y)
        return X, y
    else:
        return X

    return df_patient



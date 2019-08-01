"""
Jul. 28, 2019
This script contains data processing utilities for
the regression setting.

For the classification task, training instances are indexed using
AssessmentIDs, and for the regression task, trainining instances
are indexed using PatientIDs.
"""
import sys
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
    """
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
):
    """
    Reshape dataset so that each row of the new dataset corresponds to one
    Patient instead of one Assessment.
    Args:
        df_assessment:
            A dataframe with rows indexed by Assessment ID.
        min_visit:
            Only consider patients with more checkpoints than min_visit.
            If the number of observations is less than min_visit for one
            patient, it might be the case that this patient dropped out eariler.
    Returns:
        df_patient:
            A dataframe with rows indexed by Patient ID.
    """
    PANSS = [
        f"P{i}" for i in range(1, 8)
    ] + [
        f"N{i}" for i in range(1, 8)
    ] + [
        f"G{i}" for i in range(1, 17)]
    if not "PatientID" in df_assessment:
        raise KeyError("df_assessment DataFame must contain a 'PatientID' column.")
    patient_ids = list(set(df_assessment["PatientID"]))

    def retrive_info(pid: int):
        return df_assessment[df_assessment["PatientID"] == pid]

    cache = list()
    for pid in patient_ids:
        patient_info = retrive_info(pid)
        cache.append(len(patient_info))
    return cache

    return df_patient



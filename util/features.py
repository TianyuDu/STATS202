"""
Contains methods to create extra features (i.e. feature engerineering)

Jul. 28, 2019

Difference between features.py and data_proc.py:
    Methods in this package does not interact with on-disk CSV files
    directly, instead, most methods work on pd.DataFrame or np.ndarray.
"""
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing


def reduce_countries(
        df: pd.DataFrame,
        major_countries: list,
) -> pd.DataFrame:
    """
    Reduces countries dummies, so that only countries contributing
    to significant portion of the sample set are preserved, other
    countries are reduced to 'Other'
    Args:
        df: the raw dataframe, it must contains "Country" column.

        major_countries: a list of strings denoting countries to be 
            preserved, countries not in this list will be reduced
            to 'Other'.

    Returns:
        reduced: A dataframe similar to df, with countries reduced.
    """
    if "Country" not in df:
        raise KeyError("df must have 'Country' column.")
    reduced = df.copy()
    for i in range(len(df)):
        if not reduced.Country[i] in major_countries:
            reduced.at[i, "Country"] = "Other"
    return reduced


def polynomial_standardized(
        df: pd.DataFrame,
        quant_features: list,
        poly_degree: int
) -> pd.DataFrame:
    """
    Generates polynomial features on features in quant_features.
    And features in quant_features will be standardized.

    Args:
        df:
            Raw dataframe.
        quant_features:
            A list of strings indicting quantitative features which
            are going to be engerineered.
        poly_degree:
            An integer denoting the max-degree of polynomial features
            to be generated.

    Returns:
        df_extended:
            the dataframe with extended features.
        CROSS:
            A list of feature names.
    """
    df_extend = df.copy()
    perserved_features = list(set(df.columns) - set(quant_features))
    df_perserved = df_extend[perserved_features]
    df_extend.drop(columns=perserved_features, inplace=True)

    if poly_degree > 1:
        print("Generating Polynomial Features...")
        poly = preprocessing.PolynomialFeatures(degree=poly_degree)
        X_poly = poly.fit_transform(df_extend)  # this is a numpy array.
        CROSS = ["Cross_" + str(i) for i in range(X_poly.shape[1])]
    else:
        print("Polynomial degree is set to 1, no polynomial feature generated.")
        X_poly = df_extend
        CROSS = []

    print("Standardizing Data...")
    scaler0 = preprocessing.StandardScaler()
    df_extend = pd.DataFrame(
        scaler0.fit_transform(df_extend.values),
        columns=df_extend.columns)
    scaler1 = preprocessing.StandardScaler()
    X_poly = scaler1.fit_transform(X_poly)
    df_poly = pd.DataFrame(X_poly, columns=CROSS)
    df_extend = pd.concat([df_perserved, df_extend, df_poly], axis=1)
    return df_extend, CROSS

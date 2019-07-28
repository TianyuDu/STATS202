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
    for v in ["A", "B", "C", "D"]:
        df_temp = load_individual_dataset(path + f"Study_{v}.csv")
        collect.append(df_temp)
        lengths.append(len(df_temp))
    df = pd.concat(collect)
    assert len(df) == sum(lengths)
    return gen_label(df)


def gen_sup(raw: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    PANSS = [f"P{i}" for i in range(1, 8)] \
        + [f"N{i}" for i in range(1, 8)] \
        + [f"G{i}" for i in range(1, 17)]
    return raw[PANSS], raw["Alert"]

if __name__ == "__main__":
    df = load_whole("./data/")

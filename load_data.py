import pandas as pd
import numpy as np


def load_individual(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=0)
    return df


def gen_label(df: pd.DataFrame) -> pd.DataFrame:
    df_cp = df.copy()
    df_cp["Pass"] = (df_cp["LeadStatus"] == "Passed").astype(np.int)
    return df_cp


def load_whole(path: str) -> pd.DataFrame:
    collect, lengths = [], []
    for v in ["A", "B", "C", "D"]:
        df_temp = load_individual(path + f"Study_{v}.csv")
        collect.append(df_temp)
        lengths.append(len(df_temp))
    df = pd.concat(collect)
    assert len(df) == sum(lengths)
    return gen_label(df)


if __name__ == "__main__":
    df = load_whole("./data/")

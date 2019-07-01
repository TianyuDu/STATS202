import pandas as pd


def load_individual(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=0)
    return df


def load_whole(path: str) -> pd.DataFrame:
    collect, lengths = [], []
    for v in ["A", "B", "C", "D"]:
        df_temp = load_individual(path + f"Study_{v}.csv")
        collect.append(df_temp)
        lengths.append(len(df_temp))
    df = pd.concat(collect)
    assert len(df) == sum(lengths)
    return df

df = load_whole("./data/")

import pandas as pd
import load_data

if __name__ == "__main__":
    # Load the whole dataset
    df_whole = load_data.load_whole(path="./data/")
    # Save to csv
    df_whole.to_csv("./data/train.csv")
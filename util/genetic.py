import pandas as pd
import numpy as np
import tpot
from tpot import TPOTClassifier
from sklearn import model_selection

import data


def main():
    # Load the whole dataset
    df = data.load_whole(path="./data/")
    df_shuffled = df.iloc[np.random.permutation(len(df)), :]
    df["Alert"].value_counts()
    tpot_model = TPOTClassifier(
        generations=5,
        population_size=10,
        scoring="accuracy",
        verbosity=2
    )

    X, y = data.gen_sup(df)
    X = X.astype(np.float32)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=None, shuffle=True
    )

    tpot_model.fit(X_train, y_train)
    print(tpot_model.score(X_test, y_test))
    tpot_model.export("sample_pipeline.py")


if __name__ == "__main__":
    main()

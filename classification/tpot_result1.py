import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

sys.path.append("../")
import util.data_proc
import util.features


# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Average CV score on the training set was:0.8373113981642504
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.15000000000000002, n_estimators=100), step=0.3),
    GradientBoostingClassifier(learning_rate=0.01, max_depth=2, max_features=0.15000000000000002, min_samples_leaf=8, min_samples_split=17, n_estimators=100, subsample=0.25)
)

df_train = util.data_proc.load_whole(path="../data/")
print(df_train.shape)
df_test = pd.read_csv("../data/Study_E.csv", header=0)
print(df_test.shape)
# Reduced countries
major_countries = ["USA", "Russia", "Ukraine", "China", "Japan"]
df_train = util.features.reduce_countries(df_train, major_countries)
df_test = util.features.reduce_countries(df_test, major_countries)
X_train, y_train, FEATURE, PANSS = util.data_proc.gen_slp_assessment(
    df_train)
X_test = util.data_proc.parse_test_set(X_train, df_test)
print("Design_train: {}, Design_test: {}".format(X_train.shape, X_test.shape))

exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)

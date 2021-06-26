import argparse

# trainng arguments
parser = argparse.ArgumentParser(description="Perform training on model complete version")
parser.add_argument("train_dataset",
                    type=str,
                    help="Path to dataset file used to train the model")
parser.add_argument("val_dataset",
                    type=str,
                    help="Path to dataset file to validate the model")
parser.add_argument("output_name",
                    type=str,
                    help="Path to output file")
parser.add_argument("--mode",
                    type=str,
                    choices=["complete", "abridged"],
                    default="complete",
                    help="Model complete or abridged")

# parse arguments
args = vars(parser.parse_args())

print("Loading modules...")

# supress all warnings
import warnings
warnings.filterwarnings("ignore")

# Arrays and datasets
import numpy as np
import pandas as pd

# oversampling
from imblearn.over_sampling import RandomOverSampler

# modelling
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

# for reproducible results
np.random.seed(42)

# improve scikit-learn performance for Intel processors
from sklearnex import patch_sklearn
patch_sklearn()

# dataset used to train the model
print("Loading the dataset...")
df_train = pd.read_csv(args["train_dataset"])
df_test = pd.read_csv(args["val_dataset"])

# split features and label, use all data in the training set (not splitting it as we do in above code)
print("Preprocessing...")
X_train = df_train.iloc[:, 1:-1].values
y_train = df_train.iloc[:, 5].values

# do oversampling to handle imbalanced class
sampler = RandomOverSampler()
X_train, y_train = sampler.fit_resample(X_train, y_train)

# do dimension reduction
if args["mode"] == "abridged":
    pca = PCA(n_components=3)
    X_train = pca.fit_transform(X_train)

# fit the model
clf = RandomForestClassifier(n_jobs=-1)

# prepare grid search parameters
param_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=1500, num=10)],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(10, 100, num=10)] + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# create grid search pipeline
cv = HalvingGridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=2)

# run fine tuning
cv.fit(X_train, y_train)

# return the best parameters
print("Best score: ", cv.best_score_)
print("Best parameters:")
print(cv.best_params_)

# select the same column from test dataset
X_new = None
if args["mode"] == "abridged":
    X_new = pca.transform(df_test.iloc[:, 1:].values)
else:
    X_new = df_test.iloc[:, 1:].values

# run predictions, the result will be saved in y_pred as numpy array
print("Running predictions...")
y_pred = cv.best_estimator_.predict(X_new)

# save predictions
df_test["failed"] = y_pred
df_test[["job_id", "failed"]].to_csv(args["output_name"], index=None)

# get unique value percentage
print(df_test["failed"].value_counts() / len(df_test) * 100)

print("\n--- DONE ---")

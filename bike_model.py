import json
import pickle
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "hour.csv"
MODEL_FILE = BASE_DIR / "bike_rental_model.pkl"
INFO_FILE = BASE_DIR / "model_info.json"

NUMERIC_FEATURES = ["temp", "atemp", "hum", "windspeed", "yr", "holiday", "workingday"]
CATEGORICAL_FEATURES = ["season", "mnth", "hr", "weekday", "weathersit"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "cnt"


def load_data(path=DATA_FILE):
    data = pd.read_csv(path)
    return data[FEATURES], data[TARGET]


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_algorithms():
    return {
        "SVR": TransformedTargetRegressor(regressor=SVR(), transformer=StandardScaler()),
        "Ridge Regression": Ridge(),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    }


def build_model(algorithm_name):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", build_algorithms()[algorithm_name]),
    ])


def read_info():
    with open(INFO_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def train_final(algorithm_name, data_path=DATA_FILE):
    X, y = load_data(data_path)
    model = build_model(algorithm_name)
    model.fit(X, y)
    return model


def save_model(model, path=MODEL_FILE):
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path=MODEL_FILE):
    with open(path, "rb") as file:
        return pickle.load(file)


def write_info(info, path=INFO_FILE):
    info = dict(info)
    info["sklearn_version"] = sklearn.__version__
    with open(path, "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)

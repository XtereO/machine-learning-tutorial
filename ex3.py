import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def lab1():
    data = pd.read_csv("ex3_lab1.csv")
    X = data["X"].values
    print(X.mean())
    X = X.reshape(len(X), 1)
    Y = data["Y"].values
    print(Y.mean())
    Y = Y.reshape(len(Y), 1)
    model = LinearRegression().fit(X, Y)
    print(model.coef_, model.intercept_, model.score(X, Y))


def lab2(part, prepared):
    _data = pd.read_csv("fish_train.csv")
    if (prepared):
        _data["Length1"] = _data["Length1"]**3
        _data["Length2"] = _data["Length2"]**3
        _data["Length3"] = _data["Length3"]**3
        _data["Height"] = _data["Height"]**3
        _data["Width"] = _data["Width"]**3

    data = _data.drop("Species", axis=1)
    X = data.drop("Weight", axis=1)
    y = data["Weight"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=41, test_size=0.2, stratify=_data["Species"])
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    if (part == 1):
        model = LinearRegression().fit(X_train, y_train)
        print(model.score(X_test, y_test), r2_score(
            y_test, model.predict(X_test)))
        new_data = pd.read_csv("fish_reserved.csv")
        if (prepared):
            new_data["Length1"] = new_data["Length1"]**3
            new_data["Length2"] = new_data["Length2"]**3
            new_data["Length3"] = new_data["Length3"]**3
            new_data["Height"] = new_data["Height"]**3
            new_data["Width"] = new_data["Width"]**3
        new_data = new_data.drop("Species", axis=1)
        print(model.predict(new_data))
    if (part == 2):
        model = PCA(svd_solver="full", n_components=1).fit(X_train, y_train)
        print(model.explained_variance_ratio_, model.components_)
        importances = np.abs(
            model.components_ * np.sqrt(model.explained_variance_ratio_)[:, np.newaxis])
        importance_score = np.sum(importances, axis=0)
        print(importance_score)


lab2(1, True)

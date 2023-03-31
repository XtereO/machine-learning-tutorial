import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier


def lab1(p):
    data = pd.read_csv("ex4_lab1.csv")
    X = data[["X", "Y"]]
    y = data[["Class"]]
    model = kNN(n_neighbors=3, p=p).fit(X, y)

    new_point = np.array([[34, 28]])
    print(model.kneighbors(new_point), model.predict(new_point))


def load_data(path):
    data = pd.read_csv(path).drop(
        "marital-status", axis=1).drop("education", axis=1)
    data_selected = data
    print("all columns", data.shape)
    print("only numeric", data_selected.shape)
    # print("count class 0", data[data["label"] == 0].shape)
    data_selected.loc[data_selected["workclass"]
                      == "?", "workclass"] = "Private"
    data_selected.loc[data_selected["race"] == "?", "race"] = "White"
    data_selected.loc[data_selected["occupation"]
                      == "?", "occupation"] = "Prof-specialty"
    data_selected.loc[data_selected["relationship"]
                      == "?", "relationship"] = "Husband"
    data_selected.loc[data_selected["sex"] == "?", "sex"] = "Male"
    data_selected.loc[data_selected["native-country"]
                      == "?", "native-country"] = "United-States"
    data_filtrated = data_selected[(data_selected["workclass"] != "?") & (data_selected["occupation"] != "?") & (
        data_selected["relationship"] != "?") & (data_selected["race"] != "?") & (data_selected["sex"] != "?") & (data_selected["native-country"] != "?")]
    print("count missed values",
          data_filtrated.shape)
    data_dumm = pd.get_dummies(
        data_filtrated, drop_first=True)
    print("converted sample", data_dumm.shape)
    return data_dumm


def lab2(prepared):
    data = load_data("adult_data_train.csv")
    # sns.histplot(data=data_selected, x="native-country")
    # plt.show()

    X = data.drop("label", axis=1)
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=21, test_size=0.25, stratify=data["label"])

    X_predict = load_data("adult_data_reserved.csv")
    X_predict['native-country_Holand-Netherlands'] = np.zeros(
        (6513,), dtype=int)
    X_predict = X_predict[X_train.columns]

    scaler = MinMaxScaler().fit(pd.concat(objs=[X_train, X_predict]))
    if (prepared):
        X_train = pd.DataFrame(scaler.transform(
            X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(
            X_test), columns=X_test.columns, index=X_test.index)

    print("fnlwgt mean", X_train["fnlwgt"].mean())

    model = RandomForestClassifier().fit(X_train, y_train)
    print("f1_score", model.score(X_test, y_test),
          f1_score(y_test, model.predict(X_test)))

    X_predict = pd.DataFrame(scaler.transform(
        X_predict), columns=X_predict.columns, index=X_predict.index)
    result = model.predict(X_predict)
    print("prediction", result)
    file = open("ex4_lab2_results.txt", mode="w")
    for i in result:
        file.write(f", {i}")
    file.close()


lab2(True)

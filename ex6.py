from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score


def train():
    data = pd.read_csv("titanic_train.csv").drop(
        ["ticket", "cabin", "home.dest"], axis=1)
    print("missed age count", data["age"].isna().sum())
    print("count survived", data["survived"].sum(), data.shape)
    for header in data.columns:
        count_missed = data[header].isna().sum()
        print("procent exist columns", header, count_missed/981)

    data["fam_size"] = data["sibsp"] + data["parch"]
    data = data.drop(["parch", "sibsp"], axis=1)
    print("mean fam_size", data["fam_size"].mean())

    print("columns", data.columns.shape)
    print("probability based on statistics",
          data[(data["sex"] == "female") & (data["pclass"] == 1) & (data["survived"] == 1)].shape)

    data_num = data.select_dtypes(include='number')
    data_num_dropped_nan = data_num.dropna()
    data_num_X = data_num_dropped_nan.drop("survived", axis=1)
    data_num_y = data_num_dropped_nan["survived"]
    print("columns", data_num_X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        data_num_X, data_num_y, test_size=0.2, random_state=7, stratify=data_num_y)

    model = LogisticRegression(
        random_state=7, max_iter=1000).fit(X_train, y_train)
    print("f1_score num_only with droped values",
          f1_score(y_test, model.predict(X_test)))

    data_num.loc[data_num["pclass"].isna(
    ), "pclass"] = data_num_X["pclass"].mean()
    data_num.loc[data_num["age"].isna(), "age"] = data_num_X["age"].mean()
    data_num.loc[data_num["fare"].isna(), "fare"] = data_num_X["fare"].mean()

    X_train, X_test, y_train, y_test = train_test_split(
        data_num.drop("survived", axis=1), data_num["survived"], test_size=0.2, random_state=7, stratify=data_num["survived"])

    model = LogisticRegression(
        random_state=7, max_iter=1000).fit(X_train, y_train)
    print("f1_score num_only with mean values",
          f1_score(y_test, model.predict(X_test)))

    data_num_name = data.select_dtypes(include='number')
    data_num_name["name"] = data["name"]
    data_num_name["honorific"] = data["name"]
    for i in range(len(data["name"])):
        data_num_name["honorific"][i] = data["name"][i].split(", ")[
            1].split(".")[0]
    print(len(set(data_num_name["honorific"])))

    for i in range(len(data_num_name["honorific"])):
        if (data_num_name["honorific"][i] in ["Rev", "Col", "Dr", "Major", "Don", "Capt"]):
            data_num_name["honorific"][i] = "Mr"
        if (data_num_name["honorific"][i] in ["Dona", "the Countess"]):
            data_num_name["honorific"][i] = "Mrs"
        if (data_num_name["honorific"][i] in ["Mlle", "Ms"]):
            data_num_name["honorific"][i] = "Miss"
    print("mean age of Miss",
          data_num_name[data_num_name["honorific"] == "Miss"]["age"].mean())
    print("mean honorific", data_num_name["honorific"].value_counts())

    data_num_name.loc[data_num_name["age"].isna(),
                      "age"] = data_num_name[data_num_name["honorific"] == "Mr"]["age"].mean()
    print(data_num_name.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        data_num_name.drop(["survived", "honorific", "name"], axis=1), data_num_name["survived"], test_size=0.2, random_state=7, stratify=data_num_name["survived"])

    model = LogisticRegression(
        random_state=7, max_iter=1000).fit(X_train, y_train)
    print("f1_score num_only with honorific",
          f1_score(y_test, model.predict(X_test)))

    data_num_name["sex"] = data["sex"]
    data_num_name["embarked"] = data["embarked"]
    data_num_name = data_num_name.drop(["honorific", "name"], axis=1)
    data_dumm = pd.get_dummies(data_num_name, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        data_dumm.drop(["survived"], axis=1), data_dumm["survived"], test_size=0.2, random_state=7, stratify=data_dumm["survived"])

    model = LogisticRegression(
        random_state=7, max_iter=1000).fit(X_train, y_train)
    print("f1_score num_only with one-hot coding",
          f1_score(y_test, model.predict(X_test)))

    data_predict = pd.read_csv("titanic_reserved.csv").drop(
        ["ticket", "cabin", "home.dest"], axis=1)
    data_predict["fam_size"] = data_predict["sibsp"] + data_predict["parch"]
    data_predict = data_predict.drop(["parch", "sibsp"], axis=1)
    data_predict["name"] = data_predict["name"]
    data_predict["honorific"] = data_predict["name"]
    for i in range(len(data_predict["name"])):
        data_predict["honorific"][i] = data_predict["name"][i].split(", ")[
            1].split(".")[0]
    print(len(set(data_predict["honorific"])))

    for i in range(len(data_predict["honorific"])):
        if (data_predict["honorific"][i] in ["Rev", "Col", "Dr", "Major", "Don", "Capt", "Sir", "Jonkheer"]):
            data_predict["honorific"][i] = "Mr"
        if (data_predict["honorific"][i] in ["Dona", "the Countess", "Mme"]):
            data_predict["honorific"][i] = "Mrs"
        if (data_predict["honorific"][i] in ["Mlle", "Ms", "Lady"]):
            data_predict["honorific"][i] = "Miss"
    print("mean age of Miss",
          data_predict[data_predict["honorific"] == "Miss"]["age"].mean())
    print("mean honorific", data_predict["honorific"].value_counts())

    data_predict.loc[data_predict["age"].isna(),
                     "age"] = data_predict[data_predict["honorific"] == "Mr"]["age"].mean()
    print(data_predict.columns)

    data_predict = data_predict.drop(["honorific", "name"], axis=1)
    data_predict = pd.get_dummies(data_predict, drop_first=True)
    result = model.predict(data_predict)
    file = open("ex6_results.txt", mode="w")
    for i in result:
        file.write(f", {i}")
    file.close()


train()

# Missin value -> ohe -> scalling -> [Feature selection] -> DT


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv("train.csv")
df = df.drop(columns=["PassengerId", "Ticket", "Name", "Cabin"])

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["Survived"]), df["Survived"], test_size=0.2, random_state=42
)


trf1 = ColumnTransformer(
    [
        ("impute_Age", SimpleImputer(), [2]),
        ("impute_Embarked", SimpleImputer(strategy="most_frequent"), [6]),
    ],
    remainder="passthrough",
)
trf2 = ColumnTransformer(
    [
        (
            "ohe_sex_embarked",
            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            [1, 6],
        ),
    ],
    remainder="passthrough",
)
trf3 = ColumnTransformer([("scale", MinMaxScaler(), slice(0, 10))])

trf4 = SelectKBest(score_func=chi2, k=5)

trf5 = DecisionTreeClassifier()

pipe = Pipeline(
    [
        ("trf1", trf1),
        ("trf2", trf2),
        ("trf3", trf3),
        ("trf4", trf4),
        ("trf5", trf5),
    ]
)

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

import pickle
pickle.dump(pipe,open("pipe.pkl","wb"))
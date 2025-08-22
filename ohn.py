import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("cars.csv")
# print(df["owner"].value_counts())
pd.get_dummies(df, columns=["fuel", "owner"])
# print(pd.get_dummies(df,columns=["fuel","owner"],drop_first=True))
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 0:4], df.iloc[:, -1], test_size=0.2, random_state=42
)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop="first",sparse=False)
X_train_new = ohe.fit_transform(X_train[['fuel',"owner"]])
X_test_new = ohe.fit_transform(X_test[['fuel',"owner"]])

print(np.hstack((X_train[["brand","km_driven"]].values,X_train_new)).shape)
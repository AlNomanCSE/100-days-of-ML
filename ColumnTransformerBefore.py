import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("covid_toy.csv")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["has_covid"]), df["has_covid"], test_size=0.2, random_state=42
)

si = SimpleImputer()
X_train_fever = si.fit_transform(X_train[["fever"]])
X_test_fever = si.fit_transform(X_test[["fever"]])

# Ordernial encoder
oe = OrdinalEncoder(categories=[["Mild", "Strong"]])
X_train_cough = oe.fit_transform(X_train[["cough"]])
X_test_cough = oe.fit_transform(X_test[["cough"]])

# One hot encoding
ohe = OneHotEncoder(drop="first", sparse_output=False)
X_train_gender_city = ohe.fit_transform(X_train[["gender", "city"]])
X_test_gender_city = ohe.fit_transform(X_test[["gender", "city"]])

# Select the columns to be passed through (not transformed)
X_train_passthrough = X_train[["age"]]
X_test_passthrough = X_test[["age"]]

# Combine all the preprocessed columns for the training set
X_train_transformed = np.hstack(
    (X_train_fever, X_train_cough, X_train_gender_city, X_train_passthrough)
)

# Combine all the preprocessed columns for the testing set
X_test_transformed = np.hstack(
    (X_test_fever, X_test_cough, X_test_gender_city, X_test_passthrough)
)

print("X_train_transformed shape:", X_train_transformed.shape)
print("X_test_transformed shape:", X_test_transformed.shape)

# You can now use these transformed arrays for model training

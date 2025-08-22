import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Load the data
df = pd.read_csv("covid_toy.csv")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["has_covid"]), df["has_covid"], test_size=0.2, random_state=42
)

# Define the ColumnTransformer to handle all preprocessing steps
trasnformer = ColumnTransformer(
    transformers=[
        ("tnf1", SimpleImputer(), ["fever"]),
        ("tnf2", OrdinalEncoder(categories=[["Mild", "Strong"]]), ["cough"]),
        ("tnf3", OneHotEncoder(sparse_output=False, drop="first"), ["gender", "city"]),
    ],
    remainder="passthrough",
)

# Apply the transformations in a single step
trasnformer.fit_transform(X_train)
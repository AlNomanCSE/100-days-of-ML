import pandas as pd
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer

df = pd.read_csv("concrete_data.csv")
df.describe()

X = df.drop(columns=["Strength"])
y = df.iloc[:,-1]

X_train,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# --- Part 1: Linear Regression without scaling ---
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
r2_score(y_test,y_pred)


np.mean(cross_val_score(lr,X,y,scoring='r2'))

# --- Part 2: Linear Regression with StandardScaler ---
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled,y_train)
y_pred = lr.predict(X_test_scaled)
r2_score(y_test,y_pred)
np.mean(cross_val_score(lr,X,y,scoring='r2'))

for column in df.columns:
        # Create a new figure with two subplots side-by-side
        plt.figure(figsize=(14, 6))

        # Subplot 1: Histogram (Distplot)
        plt.subplot(1, 2, 1)
        sns.histplot(df[column], kde=True, bins=20)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        # Subplot 2: Q-Q Plot
        plt.subplot(1, 2, 2)
        stats.probplot(df[column], dist="norm", plot=plt)
        plt.title(f"Q-Q Plot of {column}")

        # Adjust layout and display the plots
        plt.tight_layout()
        plt.show()

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='box-cox')
X_train_transformed = pt.fit_transform(X_train+0.000001)
X_test_transformed = pt.transform(X_test+0.000001)

lr = LinearRegression()
lr.fit(X_train_transformed,y_train)
y_pred = lr.predict(X_test_transformed)
r2_score(y_test,y_pred)


np.mean(cross_val_score(lr,X_train_transformed,y_train,scoring='r2'))

pt = PowerTransformer()
X_train_transformed = pt.fit_transform(X_train)
X_test_transformed = pt.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_transformed,y_train)
y_pred = lr.predict(X_test_transformed)
r2_score(y_test,y_pred)

np.mean(cross_val_score(lr,X_train_transformed,y_train,scoring='r2'))


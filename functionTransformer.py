import pandas as pd
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

df = pd.read_csv("train.csv", usecols=["Age", "Fare", "Survived"])
df.isnull().sum()
df["Age"].fillna(df["Age"].mean(), inplace=True)
X = df.iloc[:, 1:3]
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
plt.figure(figsize=(14, 6))

# Subplot 1: Histogram to visualize the distribution
plt.subplot(1, 2, 1)
sns.histplot(X_train["Age"], kde=True, bins=50)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

# Subplot 2: Q-Q Plot to check for normality
plt.subplot(1, 2, 2)
stats.probplot(X_train["Age"], dist="norm", plot=plt)
plt.title("Q-Q Plot of Age")

plt.tight_layout()
plt.show()
plt.figure(figsize=(14, 6))

# Subplot 1: Histogram to visualize the distribution
plt.subplot(1, 2, 1)
sns.histplot(X_train["Fare"], kde=True, bins=50)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")

# Subplot 2: Q-Q Plot to check for normality
plt.subplot(1, 2, 2)
stats.probplot(X_train["Fare"], dist="norm", plot=plt)
plt.title("Q-Q Plot of Fare")

plt.tight_layout()
plt.show()

clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier()
clf_1.fit(X_train, y_train)
clf_2.fit(X_train, y_train)
y_pred_1 = clf_1.predict(X_test)
y_pred_2 = clf_2.predict(X_test)

print(f"LR : {accuracy_score(y_test,y_pred_1)}")
print(f"DT : {accuracy_score(y_test,y_pred_2)}")

trf = FunctionTransformer(func=np.log1p)
X_train_transformed = trf.fit_transform(X_train)
X_test_transformed = trf.fit_transform(X_test)
clf_1.fit(X_train_transformed, y_train)
clf_2.fit(X_train_transformed, y_train)
y_pred_1 = clf_1.predict(X_test_transformed)
y_pred_2 = clf_2.predict(X_test_transformed)
print(f"LR : {accuracy_score(y_test,y_pred_1)}")
print(f"DT : {accuracy_score(y_test,y_pred_2)}")
X_transformed = trf.fit_transform(X)
clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier()
print(
    "LR: ", np.mean(cross_val_score(clf_1, X_transformed, y, cv=10, scoring="accuracy"))
)
print(
    "DT: ", np.mean(cross_val_score(clf_2, X_transformed, y, cv=10, scoring="accuracy"))
)


plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
stats.probplot(X_train["Fare"], dist="norm", plot=plt)
plt.title("Q-Q Plot of Fare")

plt.subplot(1, 2, 2)
stats.probplot(X_train_transformed["Fare"], dist="norm", plot=plt)
plt.title("Q-Q Plot of Fare")

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
stats.probplot(X_train["Age"], dist="norm", plot=plt)
plt.title("Q-Q Plot of Age")

plt.subplot(1, 2, 2)
stats.probplot(X_train_transformed["Age"], dist="norm", plot=plt)
plt.title("Q-Q Plot of Agw")

plt.tight_layout()
plt.show()


trf2 = ColumnTransformer(
    [("log", FunctionTransformer(np.log1p), ["Fare"])], remainder="passthrough"
)
X_train_transformed_2 = trf2.fit_transform(X_train)
X_test_transformed_2 = trf2.fit_transform(X_test)
clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier()
clf_1.fit(X_train_transformed_2, y_train)
clf_2.fit(X_train_transformed_2, y_train)
y_pred_1 = clf_1.predict(X_test_transformed_2)
y_pred_2 = clf_2.predict(X_test_transformed_2)
print(f"LR : {accuracy_score(y_test,y_pred_1)}")
print(f"DT : {accuracy_score(y_test,y_pred_2)}")

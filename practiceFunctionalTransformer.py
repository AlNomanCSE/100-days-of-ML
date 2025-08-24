import pandas as pd
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

df = pd.read_csv("churn_data.csv")
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Subplot 1: Histogram to visualize the distribution
plt.subplot(1, 2, 1)
sns.histplot(X_train["customer_service_calls"], kde=True, bins=50)
plt.title("customer_service_calls Distribution")
plt.xlabel("customer_service_calls")
plt.ylabel("Frequency")

# Subplot 2: Q-Q Plot to check for normality
plt.subplot(1, 2, 2)
stats.probplot(X_train["customer_service_calls"], dist="norm", plot=plt)
plt.title("Q-Q Plot of customer_service_calls")
plt.tight_layout()
# plt.show()

clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier()

clf_1.fit(X_train,y_train)
clf_2.fit(X_train,y_train)

y_prediction_1 = clf_1.predict(X_test)
y_prediction_2 = clf_2.predict(X_test)

print(f"LR : {accuracy_score(y_test,y_prediction_1)}")
print(f"DT : {accuracy_score(y_test,y_prediction_2)}")

trf2 = ColumnTransformer([
    ("log",FunctionTransformer(np.log1p),["monthly_bill"])
],remainder="passthrough")

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
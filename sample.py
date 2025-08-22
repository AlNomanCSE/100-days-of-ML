import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataframe
df = pd.read_csv("train.csv")

# Create a countplot for the 'Pclass' column
sns.countplot(x='Pclass', data=df)

# Add labels and a title to the plot
plt.title('Passenger Count by Class on the Titanic')
plt.xlabel('Passenger Class')
plt.ylabel('Count')

# Set x-axis labels to be more descriptive
plt.xticks(ticks=[0, 1, 2], labels=['1st Class', '2nd Class', '3rd Class'])

# Display the plot
# print(df["Pclass"].value_counts().plot(kind="pie",autopct="%1.1f%%"))
sns.displot(df['Age'])

from pandas_profiling import ProfileReport
# prof = ProfileReport(df)
# prof.to_file(output_file='output.html')

df = pd.read_csv("housing_train.csv")
prof = ProfileReport(df)
prof.to_file(output_file="housing.html")
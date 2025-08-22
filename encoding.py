import pandas as pd

data = {
    "age": [33, 89, 10, 98, 9, 74, 48, 39, 30],
    "gender": ["Female", "Female", "Female", "Male", "Male", "Male", "Female", "Female", "Male"],
    "review": ["Good", "Good", "Poor", "Good", "Average", "Poor", "Poor", "Good", "Average"],
    "education": ["PG", "UG", "UG", "UG", "PG", "UG", "PG", "UG", "UG"],
    "purchased": ["Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "No"]
}

df = pd.DataFrame(data)
print(df)
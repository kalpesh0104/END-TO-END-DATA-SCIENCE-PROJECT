import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import os

df = pd.read_csv("data/hospital_data.csv")

df['gender'] = df['gender'].map({'M': 0, 'F': 1})

X = df.drop("readmitted", axis=1)
y = df["readmitted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
with open("model/hospital_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved.")


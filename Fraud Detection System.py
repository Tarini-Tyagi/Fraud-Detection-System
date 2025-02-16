import pandas as pd

# Load the dataset
df = pd.read_csv("creditcard_2023.csv")

# Display basic info
#print(df.info())

# Show first few rows
#print(df.head())

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Normalize 'Amount' column
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

# Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import pickle

# Save the model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)


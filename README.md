# Fraud-Detection-System
Step 1: Get the Fraud Detection Dataset
Use the Kaggle Credit Card Fraud Detection Dataset:
Upload the dataset (creditcard.csv) to your workspace.

Step 2: Load and Explore the Data (File: Fraud Detection System.py)
Start by loading the dataset using Pandas
Key Observations:
The dataset has Amount, 28 anonymized features (V1-V28), and Class (0 = Legit, 1 = Fraud).
The dataset is imbalanced (very few fraud cases compared to normal transactions).

Step 3: Preprocess the Data
Normalize the "Amount" Feature
Drop the "Time" Column (not useful for fraud detection) [If your dataset has it]
Handle Class Imbalance using SMOTE (Synthetic Minority Over-sampling Technique)

Step 4: Train a Machine Learning Model
Use Logistic Regression for fraud classification.

Step 5: Deploy as an API Using Flask
To use your model in a real-world application, create an API with Flask.
Create a model.pkl file to store the trained model.

Step 6: (1) Create a new app.py file for the API
        (2) Run the Flask server
        (3) Test the API with a sample request - check python.py file

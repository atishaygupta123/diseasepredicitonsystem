import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("/Users/atishay/Desktop/dbms project/symptomsdiseasepred.csv")

# Perform one-hot encoding on categorical variables
df = pd.get_dummies(df, columns=['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level'])

# Split the data into features (X) and target variable (y)
X = df.drop(columns=['Disease'])
y = df['Disease']

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the full dataset
rf_classifier.fit(X, y)

# Save the trained model to a file using joblib
joblib.dump(rf_classifier, "trained_model.joblib")
print("Trained model saved as 'trained_model.joblib'")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib
import h5py

# Load the data from CSV file
data = pd.read_csv('/Users/atishay/Desktop/dbms project/encoded data files/diabetes.csv')  # Replace 'your_new_data.csv' with the path to your new CSV file

data.dropna(axis=True, inplace=True)

for column in data.columns:
    unique_values = data[column].unique()
    print(f"Unique values in column '{column}': {unique_values}")

# Perform one-hot encoding for 'gender' column
data = pd.get_dummies(data, columns=['gender'], drop_first=True)

# Split the data into features and target variable
X = data.drop(columns=['diabetes'])
y = data['diabetes']

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model to a file using joblib
joblib.dump(model, 'diabetes.pkl')

# Convert the joblib file to HDF5 format
with h5py.File('diabetes.h5', 'w') as hf:
    hf.create_dataset('model', data=open('diabetes.pkl', 'rb').read())

print("Model trained and saved successfully!")
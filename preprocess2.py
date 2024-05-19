import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv("/Users/atishay/Desktop/dbms project/data files/diabetes_prediction_dataset.csv")

# Perform encoding for categorical variables
label_mapping = {
    "gender": {"Female": 0, "Male": 1},
    "smoking_history": {"never": 0, "No Info": 1, "current": 2, "ever": 0, "former": 3, "not current": 4}
}
df.replace(label_mapping, inplace=True)

# Save the modified DataFrame to a new CSV file
df.to_csv("modified_file.csv", index=False)

print("Modified data saved to modified_file.csv")

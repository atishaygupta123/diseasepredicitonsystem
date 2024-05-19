import pandas as pd

# Read the CSV file into a DataFrame, specifying "#" as the na_values parameter
df = pd.read_csv("/Users/atishay/Desktop/dbms project/data files/breast-cancer-dataset.csv", na_values="#")

# Remove 'S/N' and 'Year' columns
df.drop(columns=["S/N", "Year"], inplace=True)

# Drop rows containing non-finite values
df.dropna(axis=0, how="any", inplace=True)

# Encode 'Breast' column
df["Breast"] = df["Breast"].map({"Right": 1, "Left": 0})

# Encode 'Breast Quadrant' column
quadrant_mapping = {
    "Upper inner": 0,
    "Upper outer": 1,
    "Lower outer": 2,
    "Lower inner": 3
}
df["Breast Quadrant"] = df["Breast Quadrant"].map(quadrant_mapping)

# Convert "Diagnosis Result" column to numerical values
diagnosis_mapping = {
    "Benign": 0,
    "Malignant": 1
}
df["Diagnosis Result"] = df["Diagnosis Result"].map(diagnosis_mapping)

# Convert relevant columns to integers
columns_to_convert = ["Breast", "Breast Quadrant", "Diagnosis Result"]
df[columns_to_convert] = df[columns_to_convert].astype(int, errors="ignore")

# Save the modified DataFrame to a new CSV file
df.to_csv("breast.csv", index=False)

print("Data saved to preprocessed_data.csv")


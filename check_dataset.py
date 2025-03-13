import pandas as pd

# Load the dataset with latin-1 encoding
df = pd.read_csv('data/raw/sentiment140.csv', encoding='latin-1')

# Print the first few rows
print("First 5 rows:")
print(df.head())

# Print column names
print("\nColumn names:")
print(df.columns.tolist())

# Print shape
print("\nDataset shape:")
print(df.shape) 
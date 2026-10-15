
import pandas as pd

# Load the hourly bike-sharing dataset
df = pd.read_csv("data/hour.csv")

# Display basic information
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types and non-null counts:")
df.info()

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# Check date range
df["dteday"] = pd.to_datetime(df["dteday"])

print("\nDate range:")
print("Start date:", df["dteday"].min())
print("End date:", df["dteday"].max())

# Understand the prediction target
print("\nBike rental demand statistics:")
print(df["cnt"].describe())

# Verify target leakage risk
leakage_check = df["casual"] + df["registered"] == df["cnt"]

print("\nTarget leakage check:")
print("Rows where casual + registered = cnt:", leakage_check.sum())
print("Total rows:", len(df))
print("Relationship holds for all rows:", leakage_check.all())

# Check hourly coverage
print("\nUnique hours:", sorted(df["hr"].unique()))


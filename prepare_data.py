
import pandas as pd

# Load the dataset
df = pd.read_csv("data/hour.csv")

# Define the target and input features
target = "cnt"

features = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]

# Create input features (X) and target (y)
X = df[features]
y = df[target]

print("Input features shape:", X.shape)
print("Target shape:", y.shape)

print("\nSelected features:")
print(X.columns.tolist())

print("\nTarget preview:")
print(y.head())

print("\nMissing values in features:", X.isnull().sum().sum())
print("Missing values in target:", y.isnull().sum())


# Time-based train/test split

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining period:")
print(df["dteday"].iloc[0], "to", df["dteday"].iloc[split_index - 1])

print("\nTesting period:")
print(df["dteday"].iloc[split_index], "to", df["dteday"].iloc[-1])

# Verify chronological separation
print("\nTraining comes before testing:",
      df["dteday"].iloc[split_index - 1]
      <= df["dteday"].iloc[split_index])


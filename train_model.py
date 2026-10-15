
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv("data/hour.csv")

# 2. Select input features and target
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

X = df[features]
y = df["cnt"]

# 3. Split data chronologically: first 80% train, last 20% test
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 4. Create the model
model = LinearRegression()

# 5. Train the model
model.fit(X_train, y_train)

# 6. Predict bike rentals on unseen test data
y_pred = model.predict(X_test)

# 7. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("Model: Linear Regression")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nModel Evaluation:")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# 8. Compare actual and predicted values
comparison = pd.DataFrame({
    "Actual Rentals": y_test.iloc[:10].values,
    "Predicted Rentals": y_pred[:10].round(2),
})

print("\nFirst 10 Predictions:")
print(comparison.to_string(index=False))

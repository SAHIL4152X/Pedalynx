
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/hour.csv")

# Features and target
features = [
    "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]

X = df[features]
y = df["cnt"]

# Chronological 80/20 split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Generate predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nPEDALYNX - PREDICTION ANALYSIS")
print("=" * 40)
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Compare actual and predicted values
plt.figure(figsize=(12, 6))

plt.plot(
    range(len(y_test)),
    y_test.to_numpy(),
    label="Actual Demand",
    linewidth=1
)

plt.plot(
    range(len(y_pred)),
    y_pred,
    label="Predicted Demand",
    linewidth=1
)

plt.title("Pedalynx - Actual vs Predicted Bike Demand")
plt.xlabel("Test Sample (Hourly)")
plt.ylabel("Number of Bike Rentals")
plt.legend()
plt.tight_layout()

plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

print("\nGraph saved as actual_vs_predicted.png")

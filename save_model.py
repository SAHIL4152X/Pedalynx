
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/hour.csv")

# Select features
features = [
    "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]

X = df[features]
y = df["cnt"]

# Train on the complete dataset for the final saved model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# Save model and feature order
joblib.dump(
    {
        "model": model,
        "features": features
    },
    "pedalynx_model.joblib"
)

print("Pedalynx model saved successfully!")
print("File: pedalynx_model.joblib")
print("Features:", len(features))


import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("data/hour.csv")

# 2. Select features and target
features = [
    "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]

X = df[features]
y = df["cnt"]

# 3. Chronological 80/20 split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 4. Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

# 5. Train and evaluate both models
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "MAE": round(mean_absolute_error(y_test, predictions), 2),
        "RMSE": round(mean_squared_error(y_test, predictions) ** 0.5, 2),
        "R2 Score": round(r2_score(y_test, predictions), 4)
    })

# 6. Display comparison
comparison = pd.DataFrame(results)

print("PEDALYNX - MODEL COMPARISON")
print("=" * 55)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("\nModel Evaluation:")
print(comparison.to_string(index=False))

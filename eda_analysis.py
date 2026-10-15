
import pandas as pd
import matplotlib.pyplot as plt

# Load the hourly dataset
df = pd.read_csv("data/hour.csv")

# Calculate average bike rentals for each hour
hourly_demand = df.groupby("hr")["cnt"].mean()

# Create the graph
plt.figure(figsize=(10, 5))
plt.plot(hourly_demand.index, hourly_demand.values, marker="o")

# Add title and labels
plt.title("Average Bike Rentals by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Number of Rentals")

# Show every hour on the x-axis
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()

# Display the graph
plt.show()


# Compare hourly demand on working and non-working days

workingday_demand = (
    df[df["workingday"] == 1]
    .groupby("hr")["cnt"]
    .mean()
)

non_workingday_demand = (
    df[df["workingday"] == 0]
    .groupby("hr")["cnt"]
    .mean()
)

# Create a comparison graph
plt.figure(figsize=(11, 6))

plt.plot(
    workingday_demand.index,
    workingday_demand.values,
    marker="o",
    label="Working Days"
)

plt.plot(
    non_workingday_demand.index,
    non_workingday_demand.values,
    marker="s",
    label="Non-Working Days"
)

plt.title("Hourly Bike Demand: Working Days vs Non-Working Days")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Number of Rentals")

plt.xticks(range(24))
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()


# Analyse bike demand by weather condition

weather_demand = (
    df.groupby("weathersit")["cnt"]
    .mean()
)

# Create a bar chart
plt.figure(figsize=(9, 5))

plt.bar(
    weather_demand.index,
    weather_demand.values
)

plt.title("Average Bike Rentals by Weather Condition")
plt.xlabel("Weather Situation")
plt.ylabel("Average Number of Rentals")

plt.xticks([1, 2, 3, 4])
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.show()


# Check sample size and average demand by weather

weather_summary = df.groupby("weathersit")["cnt"].agg(
    ["count", "mean", "median", "std"]
)

weather_summary.columns = [
    "Number of Records",
    "Average Rentals",
    "Median Rentals",
    "Standard Deviation"
]

print("\nWeather-wise Demand Summary:")
print(weather_summary.round(2))




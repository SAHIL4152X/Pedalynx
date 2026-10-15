# Pedalynx - Bike-Sharing Demand Prediction

Pedalynx is a machine learning web application that predicts hourly bike-sharing demand using historical rental, calendar, and weather data. It uses a Random Forest Regressor and Streamlit to display predictions and model evaluation results.

## Project Overview

Bike rental demand changes according to the time of day, season, calendar conditions, and weather. Pedalynx uses historical data to estimate the total number of bike rentals for a selected hour.

## Features

- Hourly bike rental demand prediction
- Random Forest Regression model
- Actual vs. Predicted visualization
- Feature Importance chart
- Model evaluation using MAE, RMSE, and R²
- Input validation for invalid calendar combinations
- Interactive Streamlit dashboard

## Dataset

Source: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

- Dataset file: `hour.csv`
- Total records: 17,379
- Time period: 2011–2012
- Target variable: `cnt` (total hourly bike rentals)

## Features Used

The model uses these 12 features:

`season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`, `weathersit`, `temp`, `atemp`, `hum`, and `windspeed`.

The `casual` and `registered` columns are excluded from the model inputs to avoid target leakage.

## Machine Learning Workflow

1. Load and explore the dataset.
2. Check data quality and prepare the features.
3. Split the data chronologically into training and testing sets.
4. Train and compare Linear Regression and Random Forest Regression.
5. Evaluate models using MAE, RMSE, and R².
6. Save the trained model.
7. Display predictions and visualizations through Streamlit.

Training records: 13,903  
Testing records: 3,476

## Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 138.29 | 183.28 | 0.3090 |
| Random Forest Regressor | 44.99 | 69.56 | 0.9005 |

The Random Forest model achieved these results on the held-out test set:

- MAE: 44.99
- RMSE: 69.56
- R²: 0.9005

These results describe performance on the selected dataset and test period. They do not guarantee the same performance on new datasets.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

## Installation and Setup

### 1. Clone the Repository

Replace the placeholder with your actual GitHub repository URL.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Pedalynx
```

### 2. Create a Virtual Environment

For Windows Command Prompt:

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually http://localhost:8501.

## How to Use

1. Launch the Streamlit application.
2. Select the time and calendar inputs.
3. Enter the weather-related values requested by the form.
4. Generate a demand prediction.
5. Review the predicted rental count, evaluation metrics, and charts.

## Limitations

- The dataset covers 2011–2012 and may not represent current demand patterns.
- Predictions depend on historical data and the input values provided.
- Live events and real-time bike availability are not included unless represented in the dataset.
- Feature importance does not establish cause and effect.
- Evaluation metrics are based on one chronological test split.
- The project is intended for educational and portfolio purposes.

## Future Improvements

- Deploy the application publicly.
- Add automated tests and stronger input validation.
- Evaluate the model using newer data.
- Improve prediction explanations and interface guidance.
- Add model monitoring and retraining if suitable data becomes available.

## Acknowledgements

Thanks to the contributors of the UCI Bike Sharing Dataset for making it available for research and educational use.

Project: Pedalynx - Bike-Sharing Demand Prediction

Author: Add your name or GitHub profile here.

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Pedalynx | Demand Intelligence",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CYAN + BLACK THEME
# --------------------------------------------------
st.markdown("""
<style>
.stApp { background: #080D14; color: #F2FAFC; }
[data-testid="stHeader"] { background: rgba(8, 13, 20, 0.95); }
[data-testid="stSidebar"] { background: #0D141D; border-right: 1px solid #203744; }
.block-container { max-width: 1200px; padding-top: 3rem !important; padding-bottom: 3rem; overflow: visible !important; }
h1, h2, h3, h4, p, label { color: #F2FAFC; line-height: 1.5 !important; }
h1 { letter-spacing: 2px; }
div[data-testid="stMetric"] { background: #101923; border: 1px solid #23404D; padding: 20px; border-radius: 14px; }
div[data-testid="stMetric"] label { color: #A6BAC5; }
div[data-testid="stMetricValue"] { color: #00E5FF; }
.stButton > button, .stFormSubmitButton > button { background: #00E5FF; color: #061018; border: none; border-radius: 9px; font-weight: 700; min-height: 45px; transition: 0.2s; }
.stButton > button:hover, .stFormSubmitButton > button:hover { background: #73F2FF; color: #061018; border: 1px solid #73F2FF; }
div[data-baseweb="select"] > div { background: #101923; border-color: #29414D; }
hr { border-color: #203744; }
[data-testid="stForm"] { background: #0D141D; border: 1px solid #203744; border-radius: 15px; padding: 22px; }
.small-caption { color: #91A8B5; font-size: 13px; }
.pedalynx-eyebrow { display: block; color: #00E5FF !important; font-size: 13px; font-weight: 700; letter-spacing: 3px; line-height: 1.8 !important; padding: 6px 0 10px; margin: 0 0 12px; overflow: visible !important; white-space: normal; opacity: 1; }
[data-testid="stMarkdownContainer"] { overflow: visible !important; }
[data-testid="stMarkdownContainer"] p { line-height: 1.6 !important; overflow: visible !important; }
[data-testid="stAlert"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD SAVED MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("pedalynx_model.joblib")

saved_data = load_model()
model = saved_data["model"]
features = saved_data["features"]

# --------------------------------------------------
# LOAD TEST DATA AND EVALUATION
# --------------------------------------------------
@st.cache_resource
def load_evaluation_data():
    df = pd.read_csv("data/hour.csv")
    X = df[features]
    y = df["cnt"]

    # Chronological 80/20 split, matching the project's evaluation approach
    split_index = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    evaluation_model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)
    actual_values = y_test.to_numpy()

    metrics = {
        "mae": mean_absolute_error(actual_values, predictions),
        "rmse": mean_squared_error(actual_values, predictions) ** 0.5,
        "r2": r2_score(actual_values, predictions),
    }
    chart_data = pd.DataFrame({
        "Test Sample": range(len(actual_values)),
        "Actual Demand": actual_values,
        "Predicted Demand": predictions,
    })
    importance_data = pd.DataFrame({
        "Feature": features,
        "Importance": evaluation_model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    return chart_data, metrics, importance_data

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<div class="pedalynx-eyebrow">BIKE DEMAND INTELLIGENCE</div>', unsafe_allow_html=True)
st.title("🚲 PEDALYNX")
st.markdown(
    "<p class='small-caption'>Predict smarter. Ride better. "
    "Data-driven bike sharing demand forecasts.</p>",
    unsafe_allow_html=True,
)
st.divider()

try:
    chart_data, evaluation_metrics, importance_data = load_evaluation_data()
except Exception as error:
    st.error(
        "Could not load evaluation charts. Check that data/hour.csv, "
        "pedalynx_model.joblib, and required packages exist."
    )
    st.exception(error)
    st.stop()

# --------------------------------------------------
# OVERVIEW METRICS
# --------------------------------------------------
m1, m2, m3 = st.columns(3)
m1.metric("ML MODEL", "Random Forest")
m2.metric("TEST R² SCORE", f"{evaluation_metrics['r2']:.2%}")
m3.metric("INPUT FEATURES", len(features))

# --------------------------------------------------
# PREDICTION FORM
# --------------------------------------------------
st.write("")
st.subheader("Configure Your Prediction")
st.caption("Set the time, calendar and weather conditions to estimate hourly rentals.")

with st.form("prediction_form"):
    st.markdown("#### 01 — Time & Calendar")
    c1, c2, c3 = st.columns(3)

    with c1:
        season = st.selectbox(
            "Season", [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"
            }[x],
        )
        year = st.selectbox("Year", options=["2011", "2012"])

    with c2:
        month = st.selectbox(
            "Month", list(range(1, 13)), index=5,
            format_func=lambda x: [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ][x - 1],
        )
        hour = st.slider("Hour of day", 0, 23, 17)

    with c3:
        weekday = st.selectbox(
            "Day of week", list(range(7)),
            format_func=lambda x: [
                "Sunday", "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday"
            ][x],
        )
        holiday = st.selectbox(
            "Holiday?", [0, 1], format_func=lambda x: "Yes" if x else "No"
        )

    workingday = st.selectbox(
        "Working day?", [0, 1], format_func=lambda x: "Yes" if x else "No"
    )

    st.divider()
    st.markdown("#### 02 — Weather Conditions")
    w1, w2 = st.columns(2)

    with w1:
        weather = st.selectbox(
            "Weather situation", [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Clear / Mostly clear",
                2: "Mist / Cloudy",
                3: "Light rain or snow",
                4: "Heavy rain or snow",
            }[x],
        )
        temperature = st.slider("Temperature (normalized)", 0.0, 1.0, 0.5, 0.01)
        feels_like = st.slider("Feels-like temperature (normalized)", 0.0, 1.0, 0.5, 0.01)

    with w2:
        humidity = st.slider("Humidity (normalized)", 0.0, 1.0, 0.5, 0.01)
        windspeed = st.slider("Wind speed (normalized)", 0.0, 1.0, 0.2, 0.01)
        st.caption(
            "Weather values use the dataset's normalized scale (0–1), "
            "not degrees Celsius or km/h."
        )

    submitted = st.form_submit_button("⚡ PREDICT BIKE DEMAND", use_container_width=True)

# --------------------------------------------------
# INPUT VALIDATION AND PREDICTION
# --------------------------------------------------
if submitted:
    # In this dataset, a working day cannot also be a holiday.
    if holiday == 1 and workingday == 1:
        st.error(
            "Inconsistent calendar inputs: a holiday cannot be marked as a "
            "working day in this dataset. Set either Holiday? or Working day? to No, "
            "then submit again."
        )
    else:
        input_data = pd.DataFrame([{
            "season": season,
            "yr": int(year) - 2011,
            "mnth": month,
            "hr": hour,
            "holiday": holiday,
            "weekday": weekday,
            "workingday": workingday,
            "weathersit": weather,
            "temp": temperature,
            "atemp": feels_like,
            "hum": humidity,
            "windspeed": windspeed,
        }])[features]

        prediction = max(0, round(float(model.predict(input_data)[0])))
        st.divider()
        st.subheader("Prediction Result")
        result_col, info_col = st.columns([1, 2])

        with result_col:
            st.metric("ESTIMATED HOURLY RENTALS", f"{prediction:,} bikes")
        with info_col:
            st.markdown("""
            <div style="background:#0D2529;border:1px solid #00AFC2;
            border-left:4px solid #00E5FF;border-radius:10px;padding:18px;
            color:#00E5FF;font-weight:600;">
            ✓ Prediction generated successfully.
            </div>
            """, unsafe_allow_html=True)
            st.caption(
                "This is a model estimate, not a guaranteed rental count. "
                "Accuracy can vary with real-world conditions."
            )

# --------------------------------------------------
# ACTUAL VS PREDICTED CHART
# --------------------------------------------------
st.divider()
st.subheader("Actual vs Predicted Bike Demand")
st.caption(
    "Historical test-set comparison: cyan represents actual rentals, "
    "orange represents Random Forest predictions."
)
chart_col, metrics_col = st.columns([2, 1])

with chart_col:
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0D141D")
    ax.set_facecolor("#0D141D")
    display_data = chart_data.iloc[::10]
    ax.plot(display_data["Test Sample"], display_data["Actual Demand"],
            color="#00E5FF", linewidth=1.5, label="Actual Demand")
    ax.plot(display_data["Test Sample"], display_data["Predicted Demand"],
            color="#FFB454", linewidth=1.5, alpha=0.9, label="Predicted Demand")
    ax.set_xlabel("Test Sample (hourly)", color="#F2FAFC")
    ax.set_ylabel("Bike Rentals", color="#F2FAFC")
    ax.set_title("Actual vs Predicted Demand", color="#F2FAFC", pad=15)
    ax.tick_params(colors="#F2FAFC")
    ax.grid(True, alpha=0.15, color="#A6BAC5")
    for spine in ax.spines.values():
        spine.set_color("#29414D")
    legend = ax.legend(facecolor="#101923", edgecolor="#29414D")
    for item in legend.get_texts():
        item.set_color("#F2FAFC")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with metrics_col:
    st.markdown("#### Test Performance")
    st.metric("MAE", f"{evaluation_metrics['mae']:.2f} bikes")
    st.metric("RMSE", f"{evaluation_metrics['rmse']:.2f} bikes")
    st.metric("R² Score", f"{evaluation_metrics['r2']:.4f}")
    st.caption(
        "MAE and RMSE measure prediction error. Lower is generally better. "
        "R² closer to 1 indicates a better fit on this test set."
    )

st.markdown("""
<div style="background:#101923;border:1px solid #203744;border-radius:10px;
padding:14px;color:#A6BAC5;font-size:13px;">
Note: This chart evaluates the model on held-out historical data. It is not a live record of current bike rentals.
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------
st.divider()
st.subheader("What Influences Bike Demand?")
st.caption(
    "Relative feature importance from the Random Forest model. Higher values "
    "indicate greater contribution to the model's predictions; this does not prove causation."
)
importance_col, explanation_col = st.columns([2, 1])

with importance_col:
    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    fig_imp.patch.set_facecolor("#0D141D")
    ax_imp.set_facecolor("#0D141D")
    ax_imp.barh(importance_data["Feature"], importance_data["Importance"], color="#00D8EA")
    ax_imp.set_xlabel("Relative Importance", color="#F2FAFC")
    ax_imp.set_ylabel("Feature", color="#F2FAFC")
    ax_imp.set_title("Random Forest Feature Importance", color="#F2FAFC", pad=12)
    ax_imp.tick_params(colors="#F2FAFC")
    ax_imp.grid(axis="x", alpha=0.15, color="#A6BAC5")
    for spine in ax_imp.spines.values():
        spine.set_color("#29414D")
    fig_imp.tight_layout()
    st.pyplot(fig_imp, use_container_width=True)
    plt.close(fig_imp)

with explanation_col:
    st.markdown("#### How to read this chart")
    st.markdown(
        "- **Longer bar:** the model relied more on that feature.\n"
        "- **Shorter bar:** the feature had less relative importance in this model.\n"
        "- Importance is model-specific and does not mean the feature directly causes demand to change."
    )
    top_feature = importance_data.iloc[-1]
    st.metric("TOP FEATURE", top_feature["Feature"])
    st.caption(f"Relative importance: {top_feature['Importance']:.1%}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.markdown(
    '<div class="pedalynx-eyebrow">BIKE DEMAND INTELLIGENCE</div>',
    unsafe_allow_html=True,
)

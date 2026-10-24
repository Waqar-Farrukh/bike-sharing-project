import warnings

import altair as alt
import pandas as pd
import sklearn
import streamlit as st

from bike_model import (
    DATA_FILE,
    FEATURES,
    load_model,
    read_info,
    save_model,
    train_final,
)

st.set_page_config(page_title="Bike Rental Predictor", page_icon="🚲", layout="centered")

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
WEATHER = {
    1: "Clear, few clouds, partly cloudy",
    2: "Mist and cloudy",
    3: "Light rain or light snow",
    4: "Heavy rain, snow or fog",
}
SEASONS = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}


@st.cache_data
def month_to_season():
    data = pd.read_csv(DATA_FILE, usecols=["mnth", "season"])
    return data.groupby("mnth")["season"].agg(lambda s: int(s.mode().iloc[0])).to_dict()


@st.cache_resource(show_spinner="Preparing the model...")
def get_model():
    info = read_info()
    if info.get("sklearn_version") == sklearn.__version__:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                model = load_model()
            probe = pd.DataFrame([[0.5, 0.5, 0.5, 0.2, 1, 0, 1, 1, 6, 12, 3, 1]], columns=FEATURES)
            model.predict(probe)
            return model, info, False
        except Exception:
            pass
    model = train_final(info["best_algorithm"])
    try:
        save_model(model)
    except OSError:
        pass
    return model, info, True


def build_row(year, month, hour, weekday, holiday, weather, temp_c, feels_c, humidity, wind):
    season = month_to_season()[month]
    working_day = int(weekday in (1, 2, 3, 4, 5) and holiday == 0)
    return pd.DataFrame([{
        "temp": temp_c / 41,
        "atemp": feels_c / 50,
        "hum": humidity / 100,
        "windspeed": wind / 67,
        "yr": 1 if year == 2012 else 0,
        "holiday": holiday,
        "workingday": working_day,
        "season": season,
        "mnth": month,
        "hr": hour,
        "weekday": weekday,
        "weathersit": weather,
    }])[FEATURES]


def predict_count(model, row):
    return max(0, round(float(model.predict(row)[0])))


model, info, rebuilt = get_model()

st.title("🚲 Bike Rental Predictor")
st.write(
    "Predicts the number of bikes rented in one hour in Washington D.C. "
    "from the weather and the calendar. Trained on the Capital Bikeshare data of 2011 and 2012."
)

with st.form("inputs"):
    left, right = st.columns(2)
    with left:
        year = st.selectbox("Year", [2012, 2011])
        month = st.selectbox("Month", list(range(1, 13)), index=6, format_func=lambda m: MONTHS[m - 1])
        weekday = st.selectbox("Day of the week", list(range(7)), index=3, format_func=lambda d: WEEKDAYS[d])
        hour = st.slider("Hour of the day", 0, 23, 17)
        holiday = st.radio("Public holiday?", [0, 1], format_func=lambda v: "Yes" if v else "No", horizontal=True)
    with right:
        weather = st.selectbox("Weather", list(WEATHER.keys()), format_func=lambda w: WEATHER[w])
        temp_c = st.slider("Temperature (°C)", 0.0, 41.0, 28.0, 0.5)
        feels_c = st.slider("Feels-like temperature (°C)", 0.0, 50.0, 30.0, 0.5)
        humidity = st.slider("Humidity (%)", 0, 100, 55)
        wind = st.slider("Wind speed (km/h)", 0.0, 67.0, 12.0, 0.5)
    submitted = st.form_submit_button("Predict rentals", type="primary")

row = build_row(year, month, hour, weekday, holiday, weather, temp_c, feels_c, humidity, wind)
predicted = predict_count(model, row)
rmse = info["cv_mean_rmse"]

if submitted:
    st.metric("Predicted bike rentals in this hour", f"{predicted:,}")
    st.caption(
        f"Typical error of this model: about ±{rmse:,.0f} bikes per hour "
        f"(10-fold cross-validation RMSE of {info['best_algorithm']})."
    )
    season_name = SEASONS[month_to_season()[month]]
    working = "working day" if row.loc[0, "workingday"] == 1 else "non-working day"
    st.write(f"Season used: **{season_name}** · Day type: **{working}**")
    st.caption("The season is taken from the month. The dataset starts each season around the 20th-23rd of March, June, September and December.")

    hours = list(range(24))
    day_rows = pd.concat(
        [build_row(year, month, h, weekday, holiday, weather, temp_c, feels_c, humidity, wind) for h in hours],
        ignore_index=True,
    )
    day_predictions = [max(0, round(float(v))) for v in model.predict(day_rows)]
    chart_frame = pd.DataFrame({"Hour": hours, "Predicted rentals": day_predictions})
    selected_frame = pd.DataFrame({"Hour": [hour], "Predicted rentals": [predicted]})
    top = max(day_predictions) * 1.15 + 10
    line = (
        alt.Chart(chart_frame)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Hour:Q",
                title="Hour of the day",
                scale=alt.Scale(domain=[0, 23], nice=False),
                axis=alt.Axis(values=list(range(0, 24, 2))),
            ),
            y=alt.Y(
                "Predicted rentals:Q",
                title="Predicted rentals",
                scale=alt.Scale(domain=[0, top], nice=False),
            ),
            tooltip=["Hour", "Predicted rentals"],
        )
    )
    selected = (
        alt.Chart(selected_frame)
        .mark_point(size=160, filled=True, color="#ff4b4b")
        .encode(x="Hour:Q", y="Predicted rentals:Q", tooltip=["Hour", "Predicted rentals"])
    )
    st.subheader("Predicted rentals across the whole day")
    try:
        st.altair_chart(line + selected, width="stretch")
    except TypeError:
        st.altair_chart(line + selected, use_container_width=True)
    st.caption("Same conditions for every hour, only the hour changes. The red dot is the hour you selected.")

with st.expander("About the model"):
    st.write(f"**Algorithm:** {info['best_algorithm']}")
    st.write(f"**Cross-validation RMSE:** {info['cv_mean_rmse']:,.2f} ± {info['cv_std_rmse']:,.2f} bikes per hour ({info['folds']}-fold, {info['rows']:,} hours)")
    comparison = pd.DataFrame(
        {"Mean RMSE (bikes/hour)": info["all_algorithms_mean_rmse"]}
    ).sort_values("Mean RMSE (bikes/hour)")
    st.write("**All 5 algorithms compared:**")
    st.dataframe(comparison.style.format("{:,.2f}"))
    if rebuilt:
        st.info("The saved model file did not match this environment, so the model was rebuilt from hour.csv.")

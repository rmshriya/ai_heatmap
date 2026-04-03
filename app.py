import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import joblib
from datetime import datetime
from scipy.sparse import hstack


st.set_page_config(page_title="SafeStride Heatmap", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


def get_hour(time_str):
    try:
        return datetime.strptime(str(time_str).strip(), "%I:%M %p").hour
    except Exception:
        return 12


def predict_scores(df: pd.DataFrame) -> pd.DataFrame:
    model, vectorizer = load_model()

    df = df.copy()
    df["ReportText"] = df["ReportText"].fillna("")
    df["HeelTap"] = df["HeelTap"].fillna(0)
    df["Hour"] = df["Time"].apply(get_hour)

    X_text = vectorizer.transform(df["ReportText"])
    X_other = df[["HeelTap", "Hour"]].fillna(0)

    X = hstack([X_text, X_other])
    preds = model.predict(X).clip(0, 1)

    df["RiskScore"] = preds
    return df


def risk_color(score: float) -> str:
    if score >= 0.8:
        return "red"
    elif score >= 0.55:
        return "orange"
    else:
        return "green"


st.title("SafeStride Incident Risk Heatmap")
st.write("Zoom in to inspect exact incident locations and click markers for details.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("data.csv")

required_cols = ["Location", "Latitude", "Longitude", "Time", "ReportText", "HeelTap"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

if "RiskScore" not in df.columns:
    df = predict_scores(df)

df = df.dropna(subset=["Latitude", "Longitude"]).copy()
df["RiskScore"] = df["RiskScore"].clip(0, 1)

st.sidebar.header("Filters")

min_score = st.sidebar.slider("Minimum Risk Score", 0.0, 1.0, 0.0, 0.05)
heel_filter = st.sidebar.selectbox("Heel Tap Filter", ["All", "HeelTap Only", "No HeelTap"])

if heel_filter == "HeelTap Only":
    df = df[df["HeelTap"] == 1]
elif heel_filter == "No HeelTap":
    df = df[df["HeelTap"] == 0]

df = df[df["RiskScore"] >= min_score]

if df.empty:
    st.warning("No incidents match the selected filters.")
    st.stop()

center_lat = df["Latitude"].mean()
center_lon = df["Longitude"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles="OpenStreetMap"
)

# Heatmap layer
heat_data = [
    [row["Latitude"], row["Longitude"], float(row["RiskScore"])]
    for _, row in df.iterrows()
]
HeatMap(
    heat_data,
    radius=25,
    blur=18,
    max_zoom=17
).add_to(m)

# Exact incident markers
marker_cluster = MarkerCluster(name="Incident Markers").add_to(m)

for _, row in df.iterrows():
    popup_html = f"""
    <b>Location:</b> {row['Location']}<br>
    <b>Time:</b> {row['Time']}<br>
    <b>HeelTap:</b> {row['HeelTap']}<br>
    <b>RiskScore:</b> {round(float(row['RiskScore']), 3)}<br>
    <b>Report:</b> {row['ReportText']}
    """

    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=6,
        color=risk_color(float(row["RiskScore"])),
        fill=True,
        fill_opacity=0.85,
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=row["Location"]
    ).add_to(marker_cluster)

folium.LayerControl().add_to(m)

col1, col2 = st.columns([3, 1])

with col1:
    map_data = st_folium(m, width=1000, height=650)

with col2:
    st.subheader("Summary")
    st.metric("Incidents Shown", len(df))
    st.metric("Average Risk", round(df["RiskScore"].mean(), 3))
    st.metric("Heel Taps", int(df["HeelTap"].sum()))

    st.subheader("Top Risk Locations")
    top_locations = (
        df.groupby("Location", as_index=False)["RiskScore"]
        .mean()
        .sort_values("RiskScore", ascending=False)
        .head(10)
    )
    st.dataframe(top_locations, use_container_width=True)

st.subheader("Incident Table")
st.dataframe(
    df[["Location", "Time", "HeelTap", "RiskScore", "ReportText"]],
    use_container_width=True
)
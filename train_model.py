import pandas as pd
import joblib
from datetime import datetime
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor


def get_hour(time_str):
    try:
        return datetime.strptime(str(time_str).strip(), "%I:%M %p").hour
    except Exception:
        return 12


df = pd.read_csv("data.csv")

required_cols = ["Location", "Latitude", "Longitude", "Time", "ReportText", "HeelTap", "RiskLabel"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Hour"] = df["Time"].apply(get_hour)

vectorizer = TfidfVectorizer(max_features=500)
X_text = vectorizer.fit_transform(df["ReportText"].fillna(""))

X_other = df[["HeelTap", "Hour"]].fillna(0)

X = hstack([X_text, X_other])
y = df["RiskLabel"]

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained successfully.")
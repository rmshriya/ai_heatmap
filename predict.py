import joblib
import pandas as pd
from datetime import datetime
from scipy.sparse import hstack


model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


def get_hour(time_str):
    try:
        return datetime.strptime(str(time_str).strip(), "%I:%M %p").hour
    except Exception:
        return 12


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Hour"] = df["Time"].apply(get_hour)

    X_text = vectorizer.transform(df["ReportText"].fillna(""))
    X_other = df[["HeelTap", "Hour"]].fillna(0)

    X = hstack([X_text, X_other])
    preds = model.predict(X)

    df["RiskScore"] = preds.clip(0, 1)
    return df


if __name__ == "__main__":
    sample = pd.DataFrame([
        {
            "Location": "Anna Nagar, Chennai",
            "Latitude": 13.085,
            "Longitude": 80.2101,
            "Time": "9:30 PM",
            "ReportText": "someone following me",
            "HeelTap": 1
        }
    ])
    out = predict_dataframe(sample)
    print(out[["Location", "RiskScore"]])
"""Train the insurance charge regression model.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "premium_model.joblib"
FEATURES = ["age", "sex", "bmi", "smoker", "children"]
TARGET = "charges"


def find_dataset():
    candidates = [ROOT / "data" / "Health_insurance.csv", ROOT / "data" / "insurance.csv", ROOT / "Health_insurance.csv", ROOT / "insurance.csv"]
    for path in candidates:
        if path.exists():
            return path
    return None


def make_demo_data(n=2500, seed=42):
    """Synthetic data solely to let the app run before a real dataset is supplied."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 65, n)
    sex = rng.choice(["male", "female"], n)
    bmi = np.clip(rng.normal(30.5, 6.2, n), 16, 53)
    smoker = rng.choice(["yes", "no"], n, p=[0.20, 0.80])
    children = rng.integers(0, 6, n)
    charges = (900 + age * 255 + bmi * 310 + children * 480 +
               (smoker == "yes") * (19000 + age * 120) +
               np.maximum(bmi - 30, 0) * np.maximum(age - 35, 0) * 22 +
               rng.normal(0, 2600, n))
    return pd.DataFrame({"age": age, "sex": sex, "bmi": bmi, "smoker": smoker,
                         "children": children, "charges": np.maximum(charges, 500)})


def normalize_data(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {"gender": "sex", "smoking_status": "smoker", "dependents": "children", "medical_cost": "charges", "premium": "charges"}
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns})
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    df = df[FEATURES + [TARGET]].dropna()
    df["sex"] = df["sex"].astype(str).str.lower().str.strip()
    df["smoker"] = df["smoker"].astype(str).str.lower().str.strip()
    df["sex"] = df["sex"].replace({"m": "male", "f": "female"})
    df["smoker"] = df["smoker"].replace({"y": "yes", "n": "no"})
    for col in ["age", "bmi", "children", "charges"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dataset = find_dataset()
    if dataset:
        df = normalize_data(pd.read_csv(dataset))
        source = f"real dataset: {dataset.name}"
        demo = False
    else:
        df = make_demo_data()
        source = "synthetic demo data (replace with your real dataset for meaningful estimates)"
        demo = True
    if len(df) < 20:
        raise ValueError("At least 20 valid rows are required to train the model.")
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker"]),
        ("numeric", StandardScaler(), ["age", "bmi", "children"]),
    ])
    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    dump(pipeline, MODEL_PATH)
    metadata = {"source": source, "demo_mode": demo, "rows": int(len(df)),
                "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
                "r2": round(float(r2_score(y_test, predictions)), 4),
                "features": FEATURES, "target": TARGET}
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Model saved to", MODEL_PATH)
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    train()

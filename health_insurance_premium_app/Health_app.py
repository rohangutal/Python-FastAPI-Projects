import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# -----------------------------------------------------------------------------
# 1. Project Paths & Model Initialization
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "premium_model.joblib"
DATASET_PATH = ROOT_DIR / "data" / "Health_insurance.csv"


def train_or_load_model():
    """Load existing model or train a high-accuracy RandomForest model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        try:
            pipeline = joblib.load(MODEL_PATH)
            return pipeline, {"status": "loaded", "source": "saved_model"}
        except Exception:
            pass

    # Load dataset or generate fallback data
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
        source = f"Dataset ({DATASET_PATH.name})"
    else:
        # Fallback synthetic dataset generation if file is missing
        rng = np.random.default_rng(42)
        n = 1000
        df = pd.DataFrame({
            "age": rng.integers(18, 65, n),
            "sex": rng.choice(["male", "female"], n),
            "bmi": np.clip(rng.normal(30.0, 6.0, n), 16, 50),
            "children": rng.integers(0, 5, n),
            "smoker": rng.choice(["yes", "no"], n, p=[0.2, 0.8]),
            "region": rng.choice(["southwest", "southeast", "northwest", "northeast"], n),
            "charges": rng.normal(13000, 8000, n)
        })
        df["charges"] = np.maximum(df["charges"], 1000)
        source = "Synthetic Fallback Data"

    X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["age", "bmi", "children"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"]),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    # Save trained model
    joblib.dump(pipeline, MODEL_PATH)

    meta = {
        "status": "trained",
        "source": source,
        "r2_score": round(float(r2), 4),
        "mae": round(float(mae), 2),
        "total_samples": len(df),
    }
    return pipeline, meta


# Train/Load model on startup
model_pipeline, model_meta = train_or_load_model()

# -----------------------------------------------------------------------------
# 2. FastAPI Application & Data Validation
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Health Insurance Premium Predictor",
    description="Predict annual medical insurance costs using FastAPI and Scikit-Learn",
    version="1.0.0",
)


class InsuranceInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age of primary beneficiary")
    sex: Literal["male", "female"] = Field(..., description="Gender")
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body Mass Index")
    children: int = Field(..., ge=0, le=10, description="Number of dependent children")
    smoker: Literal["yes", "no"] = Field(..., description="Smoking status")
    region: Literal["southwest", "southeast", "northwest", "northeast"] = Field(
        "southwest", description="Residential region in the US"
    )


def get_bmi_category(bmi: float) -> dict:
    if bmi < 18.5:
        return {"label": "Underweight", "badge": "bg-warning text-dark"}
    elif 18.5 <= bmi < 25.0:
        return {"label": "Normal Weight", "badge": "bg-success"}
    elif 25.0 <= bmi < 30.0:
        return {"label": "Overweight", "badge": "bg-info text-dark"}
    else:
        return {"label": "Obese", "badge": "bg-danger"}


@app.post("/api/predict")
def predict_premium(data: InsuranceInput):
    try:
        input_df = pd.DataFrame([data.model_dump()])
        predicted_annual = float(model_pipeline.predict(input_df)[0])
        predicted_annual = max(500.0, round(predicted_annual, 2))
        predicted_monthly = round(predicted_annual / 12.0, 2)

        bmi_info = get_bmi_category(data.bmi)

        # Risk Assessment
        risk_score = "Low Risk"
        risk_color = "success"
        if data.smoker == "yes" and data.bmi >= 30:
            risk_score = "High Risk (Smoking + Obese)"
            risk_color = "danger"
        elif data.smoker == "yes" or data.bmi >= 30 or data.age >= 50:
            risk_score = "Moderate Risk"
            risk_color = "warning"

        # Estimated savings tip
        savings_tip = None
        if data.smoker == "yes":
            # Estimate premium if non-smoker
            non_smoker_input = input_df.copy()
            non_smoker_input["smoker"] = "no"
            ns_pred = float(model_pipeline.predict(non_smoker_input)[0])
            diff = max(0, predicted_annual - ns_pred)
            savings_tip = f"Quitting smoking could save you approximately ${diff:,.2f} per year on insurance!"

        return {
            "success": True,
            "annual_premium": predicted_annual,
            "monthly_premium": predicted_monthly,
            "bmi_category": bmi_info["label"],
            "bmi_badge": bmi_info["badge"],
            "risk_score": risk_score,
            "risk_color": risk_color,
            "savings_tip": savings_tip,
            "inputs": data.model_dump(),
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})


@app.get("/api/info")
def model_info():
    return {"success": True, "metadata": model_meta}


# -----------------------------------------------------------------------------
# 3. HTML / CSS / Bootstrap Frontend
# -----------------------------------------------------------------------------
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Insurance Premium Predictor</title>
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome Icons -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #0d6efd 0%, #0a58ca 100%);
            --card-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
            min-height: 100vh;
        }
        .navbar-custom {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid #e2e8f0;
        }
        .hero-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white;
            padding: 2.5rem 0;
            border-radius: 0 0 2rem 2rem;
            margin-bottom: 2.5rem;
            box-shadow: 0 15px 30px rgba(15, 23, 42, 0.15);
        }
        .card-custom {
            border: none;
            border-radius: 1.25rem;
            box-shadow: var(--card-shadow);
            transition: var(--transition-smooth);
            background: #ffffff;
        }
        .form-label {
            font-weight: 600;
            font-size: 0.9rem;
            color: #334155;
            margin-bottom: 0.4rem;
        }
        .form-control, .form-select {
            border-radius: 0.75rem;
            padding: 0.75rem 1rem;
            border: 1.5px solid #cbd5e1;
            font-weight: 500;
        }
        .form-control:focus, .form-select:focus {
            border-color: #0d6efd;
            box-shadow: 0 0 0 4px rgba(13, 110, 253, 0.15);
        }
        .btn-primary-custom {
            background: var(--primary-gradient);
            border: none;
            border-radius: 0.75rem;
            padding: 0.85rem 1.5rem;
            font-weight: 600;
            font-size: 1rem;
            letter-spacing: 0.3px;
            box-shadow: 0 4px 14px rgba(13, 110, 253, 0.35);
            transition: var(--transition-smooth);
        }
        .btn-primary-custom:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(13, 110, 253, 0.45);
        }
        .preset-chip {
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 2rem;
            padding: 0.4rem 0.9rem;
            font-size: 0.825rem;
            font-weight: 600;
            color: #475569;
            cursor: pointer;
            transition: var(--transition-smooth);
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
        }
        .preset-chip:hover {
            background: #0d6efd;
            color: white;
            border-color: #0d6efd;
        }
        .result-display {
            background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
            border: 2px dashed #bfdbfe;
            border-radius: 1rem;
            padding: 1.75rem;
        }
        .result-price {
            font-size: 2.75rem;
            font-weight: 800;
            color: #0d6efd;
            line-height: 1.1;
        }
        .result-monthly {
            font-size: 1.2rem;
            font-weight: 600;
            color: #64748b;
        }
        .badge-custom {
            padding: 0.5rem 0.85rem;
            border-radius: 2rem;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .slider-val {
            font-weight: 700;
            color: #0d6efd;
        }
    </style>
</head>
<body>

    <!-- Navbar -->
    <nav class="navbar navbar-expand-lg navbar-custom sticky-top">
        <div class="container">
            <a class="navbar-brand fw-bold text-primary d-flex align-items-center gap-2" href="#">
                <i class="fa-solid fa-heart-pulse fa-lg"></i>
                <span>MediShield AI</span>
            </a>
            <span class="badge bg-primary-subtle text-primary border border-primary-subtle rounded-pill">
                FastAPI + ML Model
            </span>
        </div>
    </nav>

    <!-- Hero Banner -->
    <div class="hero-banner">
        <div class="container text-center">
            <h1 class="fw-bold mb-2">Health Insurance Premium Estimator</h1>
            <p class="text-light opacity-75 mb-0 fs-5">Get instant, machine-learning powered estimates for annual medical coverage</p>
        </div>
    </div>

    <!-- Main Container -->
    <div class="container pb-5">
        <!-- Presets Row -->
        <div class="row justify-content-center mb-4">
            <div class="col-lg-10">
                <div class="d-flex align-items-center gap-2 flex-wrap justify-content-center">
                    <span class="text-muted small fw-semibold me-2"><i class="fa-solid fa-wand-magic-sparkles text-warning me-1"></i> Quick Presets:</span>
                    <button class="preset-chip" onclick="applyPreset(22, 'female', 21.5, 0, 'no', 'southwest')">
                        <i class="fa-solid fa-user-graduate"></i> Young Non-Smoker
                    </button>
                    <button class="preset-chip" onclick="applyPreset(35, 'male', 28.4, 2, 'no', 'northwest')">
                        <i class="fa-solid fa-people-roof"></i> Family Parent
                    </button>
                    <button class="preset-chip" onclick="applyPreset(45, 'male', 32.5, 1, 'yes', 'southeast')">
                        <i class="fa-solid fa-smoking"></i> Smoker Adult
                    </button>
                    <button class="preset-chip" onclick="applyPreset(60, 'female', 26.0, 0, 'no', 'northeast')">
                        <i class="fa-solid fa-user-doctor"></i> Senior Non-Smoker
                    </button>
                </div>
            </div>
        </div>

        <div class="row justify-content-center g-4">
            <!-- Form Card -->
            <div class="col-lg-6">
                <div class="card card-custom p-4">
                    <h5 class="fw-bold mb-4 d-flex align-items-center gap-2 text-dark">
                        <i class="fa-solid fa-sliders text-primary"></i> Beneficiary Details
                    </h5>

                    <form id="calculatorForm" onsubmit="event.preventDefault(); calculatePremium();">
                        <!-- Age Input -->
                        <div class="mb-3">
                            <div class="d-flex justify-content-between align-items-center">
                                <label for="age" class="form-label">Age (Years)</label>
                                <span class="slider-val" id="ageVal">30</span>
                            </div>
                            <input type="range" class="form-range" id="age" min="18" max="90" value="30" oninput="document.getElementById('ageVal').innerText = this.value">
                        </div>

                        <!-- Sex & Smoker row -->
                        <div class="row g-3 mb-3">
                            <div class="col-6">
                                <label for="sex" class="form-label">Gender</label>
                                <select class="form-select" id="sex" required>
                                    <option value="male">Male</option>
                                    <option value="female" selected>Female</option>
                                </select>
                            </div>
                            <div class="col-6">
                                <label for="smoker" class="form-label">Smoking Status</label>
                                <select class="form-select" id="smoker" required>
                                    <option value="no" selected>Non-Smoker</option>
                                    <option value="yes">Smoker 🚬</option>
                                </select>
                            </div>
                        </div>

                        <!-- BMI Input -->
                        <div class="mb-3">
                            <div class="d-flex justify-content-between align-items-center">
                                <label for="bmi" class="form-label">Body Mass Index (BMI)</label>
                                <span class="slider-val" id="bmiVal">25.0</span>
                            </div>
                            <input type="range" class="form-range" id="bmi" min="15.0" max="50.0" step="0.5" value="25.0" oninput="document.getElementById('bmiVal').innerText = parseFloat(this.value).toFixed(1)">
                            <div class="form-text small text-muted">Normal BMI is between 18.5 and 24.9</div>
                        </div>

                        <!-- Children & Region row -->
                        <div class="row g-3 mb-4">
                            <div class="col-6">
                                <label for="children" class="form-label">Children / Dependents</label>
                                <input type="number" class="form-control" id="children" min="0" max="10" value="0" required>
                            </div>
                            <div class="col-6">
                                <label for="region" class="form-label">US Region</label>
                                <select class="form-select" id="region" required>
                                    <option value="southwest" selected>Southwest</option>
                                    <option value="southeast">Southeast</option>
                                    <option value="northwest">Northwest</option>
                                    <option value="northeast">Northeast</option>
                                </select>
                            </div>
                        </div>

                        <!-- Submit Button -->
                        <button type="submit" class="btn btn-primary-custom w-100 d-flex align-items-center justify-content-center gap-2" id="submitBtn">
                            <i class="fa-solid fa-calculator"></i>
                            <span>Calculate Premium</span>
                        </button>
                    </form>
                </div>
            </div>

            <!-- Result Card -->
            <div class="col-lg-5">
                <div class="card card-custom p-4 h-100 d-flex flex-column justify-content-between">
                    <div>
                        <h5 class="fw-bold mb-4 d-flex align-items-center gap-2 text-dark">
                            <i class="fa-solid fa-chart-line text-primary"></i> Estimated Premium
                        </h5>

                        <div id="initialState" class="text-center py-5">
                            <div class="mb-3 text-muted">
                                <i class="fa-solid fa-file-invoice-dollar fa-4x opacity-25"></i>
                            </div>
                            <h6 class="fw-bold text-secondary">No Estimate Generated Yet</h6>
                            <p class="text-muted small mb-0">Adjust parameters or pick a preset and click <strong>Calculate Premium</strong>.</p>
                        </div>

                        <div id="loadingState" class="text-center py-5 d-none">
                            <div class="spinner-border text-primary mb-3" style="width: 3rem; height: 3rem;" role="status"></div>
                            <p class="fw-semibold text-muted mb-0">Running Machine Learning Model...</p>
                        </div>

                        <div id="resultCard" class="d-none">
                            <div class="result-display text-center mb-4">
                                <span class="text-uppercase small fw-bold text-muted tracking-wide">Estimated Annual Cost</span>
                                <div class="result-price my-2" id="annualDisplay">$0.00</div>
                                <div class="result-monthly" id="monthlyDisplay">$0.00 / month</div>
                            </div>

                            <div class="vstack gap-3 mb-4">
                                <div class="d-flex justify-content-between align-items-center p-3 bg-light rounded-3">
                                    <span class="fw-semibold text-secondary"><i class="fa-solid fa-weight-scale me-2 text-primary"></i>BMI Category</span>
                                    <span class="badge" id="bmiBadge">Normal</span>
                                </div>
                                <div class="d-flex justify-content-between align-items-center p-3 bg-light rounded-3">
                                    <span class="fw-semibold text-secondary"><i class="fa-solid fa-shield-halved me-2 text-primary"></i>Risk Profile</span>
                                    <span class="badge" id="riskBadge">Low Risk</span>
                                </div>
                            </div>

                            <div id="savingsBox" class="alert alert-success d-none border-0 shadow-sm" role="alert">
                                <i class="fa-solid fa-lightbulb text-success me-2"></i>
                                <span id="savingsText" class="small fw-semibold"></span>
                            </div>
                        </div>
                    </div>

                    <!-- Footer Info -->
                    <div class="pt-3 border-top mt-4 text-center">
                        <span class="text-muted extra-small" style="font-size: 0.8rem;">
                            <i class="fa-solid fa-circle-check text-success me-1"></i>
                            Powered by Scikit-Learn RandomForest Pipeline
                        </span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Bootstrap 5 JS Bundle -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

    <script>
        function applyPreset(age, sex, bmi, children, smoker, region) {
            document.getElementById('age').value = age;
            document.getElementById('ageVal').innerText = age;
            document.getElementById('sex').value = sex;
            document.getElementById('bmi').value = bmi;
            document.getElementById('bmiVal').innerText = parseFloat(bmi).toFixed(1);
            document.getElementById('children').value = children;
            document.getElementById('smoker').value = smoker;
            document.getElementById('region').value = region;
            calculatePremium();
        }

        async function calculatePremium() {
            const initial = document.getElementById('initialState');
            const loading = document.getElementById('loadingState');
            const resultCard = document.getElementById('resultCard');

            initial.classList.add('d-none');
            resultCard.classList.add('d-none');
            loading.classList.remove('d-none');

            const payload = {
                age: parseInt(document.getElementById('age').value),
                sex: document.getElementById('sex').value,
                bmi: parseFloat(document.getElementById('bmi').value),
                children: parseInt(document.getElementById('children').value),
                smoker: document.getElementById('smoker').value,
                region: document.getElementById('region').value
            };

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();
                loading.classList.add('d-none');

                if (data.success) {
                    document.getElementById('annualDisplay').innerText = '$' + data.annual_premium.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
                    document.getElementById('monthlyDisplay').innerText = '$' + data.monthly_premium.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' / month';
                    
                    const bmiBadge = document.getElementById('bmiBadge');
                    bmiBadge.innerText = data.bmi_category;
                    bmiBadge.className = 'badge ' + data.bmi_badge;

                    const riskBadge = document.getElementById('riskBadge');
                    riskBadge.innerText = data.risk_score;
                    riskBadge.className = 'badge bg-' + data.risk_color;

                    const savingsBox = document.getElementById('savingsBox');
                    if (data.savings_tip) {
                        document.getElementById('savingsText').innerText = data.savings_tip;
                        savingsBox.classList.remove('d-none');
                    } else {
                        savingsBox.classList.add('d-none');
                    }

                    resultCard.classList.remove('d-none');
                } else {
                    alert('Error calculating premium: ' + (data.error || 'Unknown error'));
                    initial.classList.remove('d-none');
                }
            } catch (err) {
                loading.classList.add('d-none');
                initial.classList.remove('d-none');
                alert('Server network error. Please ensure FastAPI server is running.');
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_CONTENT


# -----------------------------------------------------------------------------
# 4. Main Execution Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("Starting Health Insurance Premium Predictor FastAPI App...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

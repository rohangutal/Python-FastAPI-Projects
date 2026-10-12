# Health Insurance Premium Prediction

A small full-stack machine-learning app using **FastAPI + Python** for the API/model and **HTML, CSS, Bootstrap** for the UI. It follows the five predictors in the supplied notebook: `age`, `sex`, `bmi`, `smoker`, and `children`, with `charges` as the target.

## Features
- Responsive premium prediction form
- FastAPI JSON endpoint at `POST /api/predict`
- Input validation and `/api/health` endpoint
- Linear Regression pipeline with categorical encoding
- Model metrics saved to `model/metadata.json`
- Uses a real CSV when supplied; synthetic demo data is used only to let the application run before a dataset is added

## Run locally (Windows)
1. Install Python 3.10 or newer.
2. Extract this folder and open a terminal in it.
3. Create and activate a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
5. Start the app:
   ```powershell
   uvicorn app.main:app --reload
   ```
6. Open **http://127.0.0.1:8000**. Interactive API docs: **http://127.0.0.1:8000/docs**.

## Use your real dataset (recommended)
Copy your `Health_insurance.csv` or `insurance.csv` into the `data/` folder. The CSV should contain columns named `age`, `sex`, `bmi`, `smoker`, `children`, and `charges`. Common variants `gender`, `smoking_status`, `dependents`, `medical_cost`, and `premium` are also recognized where applicable.

Then retrain the model from the project root:
```powershell
python train_model.py
```
Restart Uvicorn if it is already running. The app uses the real dataset when found. Dataset currency is preserved: the output is in the same currency units as the `charges` column; the app does not convert USD to INR.

## API example
`POST /api/predict`
```json
{
  "age": 30,
  "sex": "female",
  "bmi": 26.5,
  "smoker": "no",
  "children": 1
}
```

## Important limitation
This is an educational estimation tool, not an insurer quote or medical/financial advice. The fallback model uses synthetic data and must not be treated as a reliable real-world premium predictor. Train and evaluate with a suitable real dataset before presenting estimates as meaningful. The reported test metrics are a simple holdout evaluation, not a guarantee of future accuracy.

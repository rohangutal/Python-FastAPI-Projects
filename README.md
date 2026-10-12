# ⚡ Python FastAPI Projects

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-003B57.svg?style=flat&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A collection of end-to-end, production-ready **FastAPI** applications showcasing Machine Learning model deployment, full-stack RESTful CRUD architectures, and server-side rendering.

---

## 📂 Repository Structure

```text
Python-FastAPI-Projects/
├── health_insurance_premium_app/    # 🏥 ML-powered Health Insurance Predictor
│   ├── app/                         # FastAPI routes and server logic
│   ├── data/                        # Insurance dataset directory
│   ├── model/                       # Serialized ML pipeline and metadata
│   ├── Health_app.py                # Standalone UI/API execution script
│   ├── train_model.py               # Model training and evaluation script
│   └── requirements.txt             # App dependencies
│
├── student_management_fastapi/      # 🎓 Full-Stack Student Management CRUD
│   ├── static/                      # CSS styling and assets
│   ├── templates/                   # Jinja2 HTML templates
│   ├── main.py                      # FastAPI app with SQLite database
│   └── requirements.txt             # App dependencies
│
├── fastapi_intro.py                 # ⚡ Minimal starter FastAPI application
├── .gitignore                       # Standard Python ignore rules
└── README.md                        # Project documentation
```

---

## 🚀 Projects Overview

### 1. 🏥 Health Insurance Premium Predictor (`health_insurance_premium_app/`)
A machine learning web application that estimates health insurance charges based on health and demographic indicators.
* **Key Features**:
  * **ML Pipeline**: Scikit-Learn Linear Regression model trained with categorical encoding (`age`, `sex`, `bmi`, `smoker`, `children`).
  * **Interactive UI**: Responsive user interface built with Bootstrap and modern styling.
  * **REST API**: Production endpoints including `POST /api/predict` and `GET /api/health` with automated Swagger/OpenAPI docs.
  * **Resilient Architecture**: Supports real CSV datasets with seamless fallback to synthetic baseline modeling.

### 2. 🎓 Student Management System (`student_management_fastapi/`)
A server-rendered CRUD management application with database persistence.
* **Key Features**:
  * **Full RESTful API**: Endpoints for `GET`, `POST`, `PUT`, and `DELETE` operations.
  * **Database Layer**: Built-in SQLite database with automated table initialization on startup (`@asynccontextmanager lifespan`).
  * **No-JavaScript UI**: Robust server-side form submission using Jinja2 templates and FastAPI routing.
  * **Data Validation**: Request validation and serialization using Pydantic models.

### 3. ⚡ FastAPI Intro (`fastapi_intro.py`)
A lightweight introductory script demonstrating the foundational principles of FastAPI routing and ASGI Uvicorn server execution.

---

## 🛠️ Quickstart & Local Setup

### Prerequisites
- Python 3.10 or higher
- Git

### Cloning the Repository
```bash
git clone https://github.com/rohangutal/Python-FastAPI-Projects.git
cd Python-FastAPI-Projects
```

---

### Running Project 1: Health Insurance Premium Predictor
```powershell
cd health_insurance_premium_app

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # On Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain model with dataset
python train_model.py

# Launch the application
uvicorn app.main:app --reload
```
* **Web UI**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Interactive API Docs (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### Running Project 2: Student Management System
```powershell
cd ../student_management_fastapi

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1      # On Linux/macOS: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
uvicorn main:app --reload
```
* **Web UI**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Interactive API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🤝 Contributing & License
Contributions, feedback, and feature requests are welcome!  
This project is open-source and licensed under the MIT License.

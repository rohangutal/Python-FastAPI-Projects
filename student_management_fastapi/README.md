# Student Management System — Python-only frontend logic

This version uses **FastAPI + Python + SQLite + HTML + CSS**. It does not use JavaScript. Forms submit to FastAPI routes, and Python renders the pages with Jinja2 templates.

## Features
- View all students and search records
- Add, edit and delete student records using standard HTML forms
- REST API endpoints: GET, POST, PUT and DELETE
- SQLite database created automatically
- Swagger documentation at `/docs`

## Run on Windows
Open a terminal in this folder and run:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

If PowerShell prevents environment activation, use Command Prompt, or run:

```bash
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m uvicorn main:app --reload
```

## Open
- Frontend: http://127.0.0.1:8000
- Swagger API docs: http://127.0.0.1:8000/docs

## API endpoints
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/get_student` | View all students |
| GET | `/get_student/{roll}` | Get one student |
| POST | `/add_student` | Add a student (JSON API) |
| PUT | `/edit_student/{roll}` | Update a student (JSON API) |
| DELETE | `/delete_student/{roll}` | Delete a student (API) |

The browser UI uses separate server-rendered form routes under `/ui/...`. This keeps the frontend interactive without JavaScript. The API endpoints above remain available for the assignment and can be tested through Swagger.

## Files
```text
student_management_fastapi_python_only/
├── main.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
└── static/
    └── style.css
```

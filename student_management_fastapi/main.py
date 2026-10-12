from pathlib import Path
import sqlite3
from contextlib import asynccontextmanager, contextmanager

from fastapi import FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "students.db"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Student Management API", version="1.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class StudentCreate(BaseModel):
    roll: int = Field(..., gt=0)
    name: str = Field(..., min_length=1, max_length=100)
    course: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=3, max_length=150)


class StudentUpdate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    course: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=3, max_length=150)


@contextmanager
def get_db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS students (
                roll INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                course TEXT NOT NULL,
                email TEXT NOT NULL
            )
        """)




def read_students(search: str = ""):
    search = search.strip()
    with get_db() as db:
        if search:
            pattern = f"%{search}%"
            rows = db.execute("""
                SELECT roll, name, course, email FROM students
                WHERE CAST(roll AS TEXT) LIKE ? OR name LIKE ? OR course LIKE ? OR email LIKE ?
                ORDER BY roll
            """, (pattern, pattern, pattern, pattern)).fetchall()
        else:
            rows = db.execute(
                "SELECT roll, name, course, email FROM students ORDER BY roll"
            ).fetchall()
    return [dict(row) for row in rows]


def read_student(roll: int):
    with get_db() as db:
        row = db.execute(
            "SELECT roll, name, course, email FROM students WHERE roll = ?", (roll,)
        ).fetchone()
    return dict(row) if row else None


def render_home(request: Request, message: str = "", error: str = "",
                search: str = "", edit_student=None):
    students = read_students(search)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "students": students,
            "total_students": len(read_students()),
            "message": message,
            "error": error,
            "search": search,
            "edit_student": edit_student,
        },
    )


# Frontend page (server-rendered; no JavaScript required)
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home(request: Request, message: str = "", error: str = "", search: str = ""):
    return render_home(request, message, error, search)


@app.post("/ui/add", response_class=HTMLResponse, include_in_schema=False)
def ui_add_student(
    request: Request,
    roll: int = Form(...),
    name: str = Form(...),
    course: str = Form(...),
    email: str = Form(...),
):
    name, course, email = name.strip(), course.strip(), email.strip()
    if roll <= 0 or not name or not course or not email:
        return render_home(request, error="Please enter valid details.")
    try:
        with get_db() as db:
            db.execute(
                "INSERT INTO students (roll, name, course, email) VALUES (?, ?, ?, ?)",
                (roll, name, course, email)
            )
    except sqlite3.IntegrityError:
        return render_home(request, error=f"Roll number {roll} already exists.")
    return RedirectResponse(url="/?message=Student+added+successfully", status_code=303)


@app.get("/ui/edit/{roll}", response_class=HTMLResponse, include_in_schema=False)
def ui_edit_form(request: Request, roll: int):
    student = read_student(roll)
    if not student:
        return render_home(request, error="Student not found.")
    return render_home(request, edit_student=student)


@app.post("/ui/edit/{roll}", response_class=HTMLResponse, include_in_schema=False)
def ui_update_student(
    request: Request,
    roll: int,
    name: str = Form(...),
    course: str = Form(...),
    email: str = Form(...),
):
    name, course, email = name.strip(), course.strip(), email.strip()
    if not name or not course or not email:
        return render_home(request, error="Please complete every field.")
    with get_db() as db:
        cursor = db.execute(
            "UPDATE students SET name = ?, course = ?, email = ? WHERE roll = ?",
            (name, course, email, roll)
        )
        if cursor.rowcount == 0:
            return render_home(request, error="Student not found.")
    return RedirectResponse(url="/?message=Student+updated+successfully", status_code=303)


@app.post("/ui/delete/{roll}", include_in_schema=False)
def ui_delete_student(roll: int):
    with get_db() as db:
        db.execute("DELETE FROM students WHERE roll = ?", (roll,))
    return RedirectResponse(url="/?message=Student+deleted+successfully", status_code=303)


# REST API endpoints required by the assignment

@app.get("/get_student")
def get_students():
    return read_students()


@app.get("/get_student/{roll}")
def get_student(roll: int):
    student = read_student(roll)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


@app.post("/add_student", status_code=status.HTTP_201_CREATED)
def add_student(student: StudentCreate):
    try:
        with get_db() as db:
            db.execute(
                "INSERT INTO students (roll, name, course, email) VALUES (?, ?, ?, ?)",
                (student.roll, student.name.strip(), student.course.strip(), student.email.strip())
            )
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="A student with this roll number already exists")
    return {"message": "Student added successfully", "student": student.model_dump()}


@app.put("/edit_student/{roll}")
def edit_student(roll: int, student: StudentUpdate):
    with get_db() as db:
        cursor = db.execute(
            "UPDATE students SET name = ?, course = ?, email = ? WHERE roll = ?",
            (student.name.strip(), student.course.strip(), student.email.strip(), roll)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Student not found")
        row = db.execute(
            "SELECT roll, name, course, email FROM students WHERE roll = ?", (roll,)
        ).fetchone()
    return {"message": "Student updated successfully", "student": dict(row)}


@app.delete("/delete_student/{roll}")
def delete_student(roll: int):
    with get_db() as db:
        cursor = db.execute("DELETE FROM students WHERE roll = ?", (roll,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Student not found")
    return {"message": f"Student with roll number {roll} deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


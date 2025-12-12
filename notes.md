# 🚀 FastAPI Tutorial: From Zero to Database

A progressive, beginner‑friendly tutorial for building REST APIs with **FastAPI** — starting from nothing and ending with a real database.

---

# 📌 Prerequisites

* Python **3.11+** installed
* Basic Python knowledge (variables, functions, dictionaries)
* Terminal / Command-line basics

---

# 🧰 Step 1: Project Setup with UV

UV is a fast Python package manager.

## 1.1 Install UV

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## 1.2 Create Project

```bash
uv init student-api
cd student-api
uv venv
.venv\Scripts\activate
```

Project structure:

```
student-api/
├── pyproject.toml
├── README.md
├── main.py
└── .python-version
```

---

## 1.3 Verify Setup

```bash
uv --version
uv run python --version
```

✅ UV and Python versions should print.

---

# 🏎 Step 2: Install FastAPI

## 2.1 Add FastAPI

```bash
uv add fastapi[standard]
```

This installs:

* **fastapi**
* **uvicorn** (server)
* **pydantic** (validation)

## 2.2 Verify

```bash
uv run python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"
```

---

# 🧪 Step 3: First API Endpoint

## 3.1 Edit `main.py`

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello, World!"}
```

## 3.2 Run the Server

```bash
fastapi dev
# OR
uv run uvicorn main:app --reload
```

## 3.3 Test API

* [http://127.0.0.1:8000](http://127.0.0.1:8000) → JSON response
* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) → Swagger UI

---

# 👨‍🎓 Step 4: CRUD with In‑Memory Data

## 4.1 Demo Students

```python
students = [
    {"id": 1, "name": "Ali Khan", "age": 20, "email": "ali@example.com"},
    {"id": 2, "name": "Fatima Ahmed", "age": 22, "email": "fatima@example.com"},
    {"id": 3, "name": "Hassan Raza", "age": 19, "email": "hassan@example.com"},
]

@app.get("/students")
def get_all_students():
    return students
```

---

## 4.2 Get Single Student

```python
@app.get("/students/{student_id}")
def get_student(student_id: int):
    for s in students:
        if s["id"] == student_id:
            return s
    return {"error": "Student not found"}
```

---

## 4.3 Create a Student

```python
from pydantic import BaseModel

class Student(BaseModel):
    name: str
    age: int
    email: str

@app.post("/students")
def create_student(student: Student):
    return {"message": "Student created", "student": student}
```

---

# 🗄 Step 5: SQLModel Basics

## 5.1 Install SQLModel

```bash
uv add sqlmodel
```

## 5.2 Student Model

```python
from sqlmodel import SQLModel, Field

class Student(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    age: int
    email: str
```

---

# 🗃 Step 6: Database Integration (SQLite → Persistent Data)

SQLite database setup + CRUD using SQLModel:

```python
from fastapi import FastAPI, HTTPException, Depends
from sqlmodel import SQLModel, Field, Session, create_engine, select

DATABASE_URL = "sqlite:///students.db"
engine = create_engine(DATABASE_URL)

def create_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

class StudentBase(SQLModel):
    name: str
    age: int
    email: str

class Student(StudentBase, table=True):
    id: int | None = Field(default=None, primary_key=True)

class StudentCreate(StudentBase):
    pass

app = FastAPI()

@app.on_event("startup")
def on_startup():
    create_db()

@app.post("/students", response_model=Student)
def create_student(student: StudentCreate, session: Session = Depends(get_session)):
    db_student = Student.model_validate(student)
    session.add(db_student)
    session.commit()
    session.refresh(db_student)
    return db_student

@app.get("/students")
def list_students(session: Session = Depends(get_session)):
    return session.exec(select(Student)).all()

@app.get("/students/{student_id}")
def get_student(student_id: int, session: Session = Depends(get_session)):
    student = session.get(Student, student_id)
    if not student:
        raise HTTPException(404, "Student not found")
    return student

@app.put("/students/{student_id}")
def update_student(student_id: int, data: StudentCreate, session: Session = Depends(get_session)):
    student = session.get(Student, student_id)
    if not student:
        raise HTTPException(404, "Student not found")
    student.name = data.name
    student.age = data.age
    student.email = data.email
    session.commit()
    session.refresh(student)
    return student

@app.delete("/students/{student_id}")
def delete_student(student_id: int, session: Session = Depends(get_session)):
    student = session.get(Student, student_id)
    if not student:
        raise HTTPException(404, "Student not found")
    session.delete(student)
    session.commit()
    return {"message": "Student deleted"}
```

---

# ☁️ Step 7: Cloud Database with Neon (Optional)

## 7.1 Install PostgreSQL Driver

```bash
uv add psycopg2-binary python-dotenv
```

## 7.2 Use Neon Connection

`.env`:

```
DATABASE_URL=postgresql://user:pass@ep-xxx.region.aws.neon.tech/neondb?sslmode=require
```

Update connection:

```python
import os
DATABASE_URL = os.environ.get("DATABASE_URL")
engine = create_engine(DATABASE_URL, echo=True)
```

---

# 🎯 Summary

| Step | What You Learned         |
| ---- | ------------------------ |
| 1    | UV project setup         |
| 2    | Installing FastAPI       |
| 3    | First GET endpoint       |
| 4    | CRUD operations          |
| 5    | SQLModel basics          |
| 6    | Database integration     |
| 7    | Cloud database with Neon |

---

# 🚀 Next Steps

* Add validation
* Add relationships (Students → Courses)
* Add authentication
* Deploy to production

---

# 💬 Quick Prompt Examples (Claude / AI Tools)

* "Add grade field to students (optional)."
* "Validate age between 5–100 and email format."
* "Add pagination support."
* "Add filtering by name and age range."

---

# 🛠 Troubleshooting

```bash
uv sync              # Fix missing modules
lsof -i :8000        # Find process on port 8000
kill -9 <PID>        # Kill process
rm students.db       # Reset SQLite
```

---

Happy coding! 🚀🔥

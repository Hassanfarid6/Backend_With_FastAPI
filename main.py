from fastapi import FastAPI, HTTPException, Depends, Response, status
from pydantic import BaseModel
from typing import List, Optional
from sqlmodel import SQLModel, Field, create_engine, Session, select
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL is None:
    raise ValueError("DATABASE_URL environment variable not set")

engine = create_engine(DATABASE_URL, echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # on startup
    create_db_and_tables()
    # pre-populate data
    with Session(engine) as session:
        if not session.exec(select(Student)).first():
            students_to_add = [
                Student(name="Ali Khan", age=20, email="ali@example.com"),
                Student(name="Fatima Ahmed", age=22, email="fatima@example.com"),
                Student(name="Hassan Raza", age=19, email="hassan@example.com"),
            ]
            for student in students_to_add:
                session.add(student)
            session.commit()
    yield
    # on shutdown
    # any cleanup can go here

app = FastAPI(lifespan=lifespan)

class Student(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    age: int
    email: str

class CreateStudent(BaseModel):
    name: str
    age: int
    email: str

class UpdateStudent(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    email: Optional[str] = None

def get_session():
    with Session(engine) as session:
        yield session

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/students", response_model=List[Student])
def get_all_students(session: Session = Depends(get_session)):
    students = session.exec(select(Student)).all()
    return students

@app.get("/students/{student_id}", response_model=Student)
def get_student(student_id: int, session: Session = Depends(get_session)):
    student = session.get(Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student

@app.post("/students", response_model=Student, status_code=status.HTTP_201_CREATED)
def create_student(student: CreateStudent, session: Session = Depends(get_session)):
    db_student = Student.from_orm(student)
    session.add(db_student)
    session.commit()
    session.refresh(db_student)
    return db_student

@app.put("/students/{student_id}", response_model=Student)
def update_student(student_id: int, student_update: UpdateStudent, session: Session = Depends(get_session)):
    student = session.get(Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    update_data = student_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(student, key, value)
    
    session.add(student)
    session.commit()
    session.refresh(student)
    return student

@app.delete("/students/{student_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_student(student_id: int, session: Session = Depends(get_session)):
    student = session.get(Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    session.delete(student)
    session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

"""
Database initialization and session management for Patient Management System
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from pathlib import Path

# Import Base from patient models
from backend.models.patient import Base

# Create database directory
DB_DIR = Path(__file__).parent.parent / "database"
DB_DIR.mkdir(exist_ok=True)

# Database URL
DATABASE_URL = f"sqlite:///{DB_DIR}/patient_management.sqlite"

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database():
    """Initialize database tables"""
    # Import models to ensure they're registered with Base
    from backend.models import patient
    
    print(f"Initializing Patient Management database at: {DATABASE_URL}")
    Base.metadata.create_all(bind=engine)
    print("[OK] Patient Management database initialized")

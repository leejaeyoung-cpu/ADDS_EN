"""
Database initialization script for ADDS Patient Management System
Creates patients table and adds Inha Hospital test patient
"""
import sqlite3
from pathlib import Path
from datetime import datetime

# Database path
db_path = Path("backend/patients.db")
db_path.parent.mkdir(parents=True, exist_ok=True)

# Connect
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create patients table
cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    diagnosis TEXT,
    stage TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# Create ct_analyses table
cursor.execute("""
CREATE TABLE IF NOT EXISTS ct_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tumors_detected INTEGER DEFAULT 0,
    largest_tumor_mm REAL DEFAULT 0,
    volume_cm3 REAL DEFAULT 0,
    radiomics JSON,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
)
""")

# Insert Inha Hospital test patient
cursor.execute("""
INSERT OR REPLACE INTO patients (patient_id, name, age, gender, diagnosis, stage, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?)
""", ('P-INHA-2026-001', '인하대학병원 환자', 65, 'M', 'Colorectal Cancer', 'Stage III', datetime.now()))

conn.commit()

# Verify
cursor.execute("SELECT COUNT(*) FROM patients")
count = cursor.fetchone()[0]
print(f"[OK] Database initialized: {count} patients")

cursor.execute("SELECT patient_id, name FROM patients")
rows = cursor.fetchall()
print("\nPatient list:")
for row in rows:
    print(f"  - {row[0]}: {row[1]}")

conn.close()
print(f"\n[OK] Database created: {db_path.absolute()}")

import sqlite3, os, json
from pathlib import Path

dbs = [
    "F:/ADDS/database/patient_management.sqlite",
    "F:/ADDS/database/cdss_records.sqlite",
]

for db_path in dbs:
    print(f"\n=== {Path(db_path).name} ===")
    if not os.path.exists(db_path):
        print("  NOT FOUND")
        continue
    print(f"  Size: {os.path.getsize(db_path) / 1024:.1f} KB")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    print(f"  Tables: {tables}")
    for t in tables:
        count = cursor.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
        print(f"    {t}: {count} rows")
        if count > 0 and count < 20:
            cursor.execute(f"SELECT * FROM [{t}] LIMIT 3")
            cols = [d[0] for d in cursor.description]
            print(f"      Columns: {cols}")
            for row in cursor.fetchall():
                print(f"      {dict(zip(cols, row))}")
        elif count > 0:
            cursor.execute(f"SELECT * FROM [{t}] LIMIT 1")
            cols = [d[0] for d in cursor.description]
            print(f"      Columns: {cols}")
    conn.close()

# Also check for any other .sqlite or .db files
for ext in ["*.sqlite", "*.db"]:
    for p in Path("F:/ADDS/database").glob(ext):
        if str(p) not in dbs:
            print(f"\n  EXTRA DB: {p.name} ({p.stat().st_size / 1024:.1f} KB)")

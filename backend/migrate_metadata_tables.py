"""
Database migration script for metadata learning tables
Run this to create new tables in existing database
"""

from backend.models import Base
from backend.models.metadata_learning import (
    PatientMetadata,
    AnalysisResult,
    TreatmentOutcome,
    TumorDrugInteraction,
    ModelTrainingHistory,
    PerformanceMetric
)
from backend.database_init import engine
from sqlalchemy import inspect


def check_table_exists(table_name):
    """Check if table already exists"""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def migrate_metadata_tables():
    """Create new metadata learning tables"""
    print("="*80)
    print("METADATA LEARNING TABLES MIGRATION")
    print("="*80)
    
    tables_to_create = [
        ('patient_metadata', PatientMetadata),
        ('analysis_results', AnalysisResult),
        ('treatment_outcomes', TreatmentOutcome),
        ('tumor_drug_interactions', TumorDrugInteraction),
        ('model_training_history', ModelTrainingHistory),
        ('performance_metrics', PerformanceMetric)
    ]
    
    created_tables = []
    existing_tables = []
    
    for table_name, model_class in tables_to_create:
        if check_table_exists(table_name):
            print(f"[SKIP] Table '{table_name}' already exists")
            existing_tables.append(table_name)
        else:
            # Create single table
            model_class.__table__.create(bind=engine, checkfirst=True)
            print(f"[CREATE] Table '{table_name}' created successfully")
            created_tables.append(table_name)
    
    print("\n" + "="*80)
    print("MIGRATION SUMMARY")
    print("="*80)
    print(f"Created: {len(created_tables)} tables")
    if created_tables:
        for table in created_tables:
            print(f"  - {table}")
    
    print(f"\nExisting: {len(existing_tables)} tables")
    if existing_tables:
        for table in existing_tables:
            print(f"  - {table}")
    
    print("\n" + "="*80)
    print("MIGRATION COMPLETE")
    print("="*80)
    
    return created_tables, existing_tables


if __name__ == "__main__":
    try:
        created, existing = migrate_metadata_tables()
        
        if len(created) > 0:
            print(f"\n✓ Successfully created {len(created)} new tables")
            print("  Database is ready for metadata learning!")
        else:
            print("\n! All tables already exist. No changes made.")
        
    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        import traceback
        traceback.print_exc()

"""
Database connection and ORM utilities for ADDS
"""

import os
from typing import Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import NullPool
import yaml
from pathlib import Path

# Base for ORM models
Base = declarative_base()

class DatabaseManager:
    """
    Manages database connections and sessions for ADDS
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize database manager
        
        Args:
            config_path: Path to config.yaml. If None, uses default location
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.db_config = config['database']
        self.engine = None
        self.SessionLocal = None
        
    def get_connection_string(self) -> str:
        """
        Build PostgreSQL connection string
        
        Returns:
            Database connection URL
        """
        password = os.getenv('ADDS_DB_PASSWORD', '')
        
        conn_string = (
            f"postgresql://{self.db_config['user']}:{password}"
            f"@{self.db_config['host']}:{self.db_config['port']}"
            f"/{self.db_config['name']}"
        )
        
        return conn_string
    
    def initialize(self, echo: bool = False):
        """
        Initialize database engine and session factory
        
        Args:
            echo: Whether to log SQL statements
        """
        conn_string = self.get_connection_string()
        
        self.engine = create_engine(
            conn_string,
            echo=echo,
            poolclass=NullPool,  # For debugging; use QueuePool in production
            pool_pre_ping=True
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        print(f"✓ Database engine initialized: {self.db_config['host']}:{self.db_config['port']}/{self.db_config['name']}")
    
    def create_tables(self):
        """
        Create all tables defined in ORM models
        """
        if self.engine is None:
            raise RuntimeError("Database engine not initialized. Call initialize() first.")
        
        Base.metadata.create_all(bind=self.engine)
        print("✓ Database tables created successfully")
    
    def drop_tables(self):
        """
        Drop all tables (use with caution!)
        """
        if self.engine is None:
            raise RuntimeError("Database engine not initialized. Call initialize() first.")
        
        Base.metadata.drop_all(bind=self.engine)
        print("⚠ All database tables dropped")
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Context manager for database sessions
        
        Yields:
            SQLAlchemy session
        
        Example:
            with db_manager.get_session() as session:
                results = session.query(Experiment).all()
        """
        if self.SessionLocal is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def execute_schema_file(self, schema_path: Optional[str] = None):
        """
        Execute SQL schema file to set up database
        
        Args:
            schema_path: Path to .sql file. If None, uses default schema
        """
        if schema_path is None:
            schema_path = Path(__file__).parent.parent.parent / "configs" / "database_schema.sql"
        
        with open(schema_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        if self.engine is None:
            raise RuntimeError("Database engine not initialized. Call initialize() first.")
        
        # Execute schema
        with self.engine.connect() as conn:
            # Split by statement (basic approach)
            statements = [stmt.strip() for stmt in sql_script.split(';') if stmt.strip()]
            
            for stmt in statements:
                try:
                    conn.execute(stmt)
                    conn.commit()
                except Exception as e:
                    print(f"Warning executing statement: {e}")
                    # Continue with other statements
        
        print("✓ Database schema executed successfully")


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Session:
    """
    Dependency function for getting database sessions
    Useful for FastAPI or similar frameworks
    
    Yields:
        Database session
    """
    with db_manager.get_session() as session:
        yield session

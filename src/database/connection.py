"""
Database connection and session management for Legal Document Knowledge Base
"""

import os
import logging
from typing import Optional, Generator
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.engine import Engine
from .models import Base

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Database connection manager with session handling"""
    
    def __init__(self, database_url: Optional[str] = None, echo: bool = False):
        """
        Initialize database connection
        
        Args:
            database_url: Database URL. If None, uses SQLite default
            echo: Whether to echo SQL statements for debugging
        """
        if database_url is None:
            # Default to SQLite database in data directory
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'data', 'legal_knowledge_base.db'
            )
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            database_url = f"sqlite:///{db_path}"
        
        self.database_url = database_url
        self.echo = echo
        self._engine = None
        self._session_factory = None
        
    @property
    def engine(self) -> Engine:
        """Get or create database engine"""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine
    
    def _create_engine(self) -> Engine:
        """Create database engine with appropriate configuration"""
        if self.database_url.startswith('sqlite'):
            # SQLite-specific configuration
            engine = create_engine(
                self.database_url,
                echo=self.echo,
                poolclass=StaticPool,
                connect_args={
                    'check_same_thread': False,
                    'timeout': 30
                }
            )
            
            # Enable foreign key constraints for SQLite
            @event.listens_for(engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()
                
        else:
            # PostgreSQL or other database configuration
            engine = create_engine(
                self.database_url,
                echo=self.echo,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
        
        return engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """Get or create session factory"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                expire_on_commit=False
            )
        return self._session_factory
    
    def create_tables(self, drop_existing: bool = False):
        """
        Create database tables
        
        Args:
            drop_existing: Whether to drop existing tables first
        """
        try:
            if drop_existing:
                logger.info("Dropping existing tables...")
                Base.metadata.drop_all(self.engine)
            
            logger.info("Creating database tables...")
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions
        
        Yields:
            Database session
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_session_sync(self) -> Session:
        """
        Get a database session (caller responsible for closing)
        
        Returns:
            Database session
        """
        return self.session_factory()
    
    def test_connection(self) -> bool:
        """
        Test database connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_database_info(self) -> dict:
        """
        Get database information
        
        Returns:
            Dictionary with database information
        """
        try:
            with self.get_session() as session:
                # Get table information
                if self.database_url.startswith('sqlite'):
                    result = session.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                    tables = [row[0] for row in result.fetchall()]
                else:
                    result = session.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public'"
                    )
                    tables = [row[0] for row in result.fetchall()]
                
                return {
                    'database_url': self.database_url,
                    'engine_info': str(self.engine),
                    'tables': tables,
                    'table_count': len(tables)
                }
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {'error': str(e)}
    
    def close(self):
        """Close database connections"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database connections closed")


# Global database connection instance
_db_connection = None


def get_database_connection(database_url: Optional[str] = None, echo: bool = False) -> DatabaseConnection:
    """
    Get global database connection instance
    
    Args:
        database_url: Database URL
        echo: Whether to echo SQL statements
        
    Returns:
        DatabaseConnection instance
    """
    global _db_connection
    
    if _db_connection is None:
        _db_connection = DatabaseConnection(database_url, echo)
    
    return _db_connection


def init_database(database_url: Optional[str] = None, drop_existing: bool = False, echo: bool = False):
    """
    Initialize database with tables
    
    Args:
        database_url: Database URL
        drop_existing: Whether to drop existing tables
        echo: Whether to echo SQL statements
    """
    db = get_database_connection(database_url, echo)
    db.create_tables(drop_existing)
    return db
"""
Database migration scripts for Legal Document Knowledge Base
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .connection import DatabaseConnection, get_database_connection
from .models import Base

logger = logging.getLogger(__name__)


class DatabaseMigration:
    """
    Database migration manager for schema updates and data migrations
    """
    
    def __init__(self, db_connection: Optional[DatabaseConnection] = None):
        """
        Initialize migration manager
        
        Args:
            db_connection: Database connection instance
        """
        self.db = db_connection or get_database_connection()
        self.migration_table = 'schema_migrations'
    
    def initialize_migration_table(self):
        """Create migration tracking table"""
        try:
            with self.db.get_session() as session:
                session.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {self.migration_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version VARCHAR(50) NOT NULL UNIQUE,
                        description TEXT,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        execution_time REAL
                    )
                """))
                logger.info("Migration table initialized")
        except Exception as e:
            logger.error(f"Error initializing migration table: {e}")
            raise
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions"""
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text(f"SELECT version FROM {self.migration_table} ORDER BY applied_at")
                )
                return [row[0] for row in result.fetchall()]
        except Exception:
            # Migration table doesn't exist yet
            return []
    
    def apply_migration(self, version: str, description: str, migration_sql: str):
        """Apply a single migration"""
        start_time = datetime.now()
        
        try:
            with self.db.get_session() as session:
                # Check if migration already applied
                existing = session.execute(
                    text(f"SELECT version FROM {self.migration_table} WHERE version = :version"),
                    {'version': version}
                ).fetchone()
                
                if existing:
                    logger.info(f"Migration {version} already applied, skipping")
                    return
                
                # Execute migration SQL - split by semicolon and execute each statement
                statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip()]
                for i, statement in enumerate(statements):
                    if statement:
                        # Remove leading comments but keep the SQL statement
                        lines = statement.split('\n')
                        sql_lines = []
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('--'):
                                sql_lines.append(line)
                        
                        if sql_lines:
                            clean_statement = ' '.join(sql_lines)
                            logger.info(f"Executing statement {i+1}/{len(statements)}: {clean_statement[:100]}...")
                            session.execute(text(clean_statement))
                
                # Record migration
                execution_time = (datetime.now() - start_time).total_seconds()
                session.execute(
                    text(f"""
                        INSERT INTO {self.migration_table} 
                        (version, description, applied_at, execution_time)
                        VALUES (:version, :description, :applied_at, :execution_time)
                    """),
                    {
                        'version': version,
                        'description': description,
                        'applied_at': datetime.now(),
                        'execution_time': execution_time
                    }
                )
                
                logger.info(f"Applied migration {version}: {description} in {execution_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Error applying migration {version}: {e}")
            raise
    
    def run_all_migrations(self):
        """Run all pending migrations"""
        self.initialize_migration_table()
        applied_migrations = self.get_applied_migrations()
        
        migrations = self.get_migration_list()
        
        for version, description, sql in migrations:
            if version not in applied_migrations:
                logger.info(f"Applying migration {version}: {description}")
                self.apply_migration(version, description, sql)
            else:
                logger.debug(f"Migration {version} already applied")
        
        logger.info("All migrations completed")
    
    def get_migration_list(self) -> List[Tuple[str, str, str]]:
        """Get list of all migrations in order"""
        return [
            (
                "001_initial_schema",
                "Create initial database schema for legal documents",
                self._get_initial_schema_sql()
            ),
            (
                "002_add_indexes",
                "Add performance indexes for common queries",
                self._get_indexes_sql()
            ),
            (
                "003_add_full_text_search",
                "Add full-text search capabilities",
                self._get_full_text_search_sql()
            ),
            (
                "004_add_entity_relationships",
                "Add entity relationship tracking",
                self._get_entity_relationships_sql()
            )
        ]
    
    def _get_initial_schema_sql(self) -> str:
        """Get SQL for initial schema creation"""
        return """
        -- This migration is handled by SQLAlchemy models
        -- Just ensure all tables are created
        """
    
    def _get_indexes_sql(self) -> str:
        """Get SQL for performance indexes"""
        return """
        -- Document indexes
        CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type);
        CREATE INDEX IF NOT EXISTS idx_documents_case_number ON documents(case_number);
        CREATE INDEX IF NOT EXISTS idx_documents_court_name ON documents(court_name);
        CREATE INDEX IF NOT EXISTS idx_documents_date_issued ON documents(date_issued);
        CREATE INDEX IF NOT EXISTS idx_documents_confidence ON documents(confidence_score);
        CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
        CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
        
        -- Entity indexes
        CREATE INDEX IF NOT EXISTS idx_entities_type ON legal_entities(entity_type);
        CREATE INDEX IF NOT EXISTS idx_entities_normalized_name ON legal_entities(normalized_name);
        CREATE INDEX IF NOT EXISTS idx_entities_frequency ON legal_entities(frequency);
        CREATE INDEX IF NOT EXISTS idx_entities_confidence ON legal_entities(confidence_score);
        
        -- Case indexes
        CREATE INDEX IF NOT EXISTS idx_cases_number ON cases(case_number);
        CREATE INDEX IF NOT EXISTS idx_cases_type ON cases(case_type);
        CREATE INDEX IF NOT EXISTS idx_cases_court ON cases(court_name);
        CREATE INDEX IF NOT EXISTS idx_cases_decision_date ON cases(decision_date);
        CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(case_status);
        
        -- Search index
        CREATE INDEX IF NOT EXISTS idx_search_content_type ON search_index(content_type);
        CREATE INDEX IF NOT EXISTS idx_search_relevance ON search_index(relevance_score);
        CREATE INDEX IF NOT EXISTS idx_search_document ON search_index(document_id);
        
        -- Processing log indexes
        CREATE INDEX IF NOT EXISTS idx_processing_log_document ON processing_logs(document_id);
        CREATE INDEX IF NOT EXISTS idx_processing_log_status ON processing_logs(status);
        CREATE INDEX IF NOT EXISTS idx_processing_log_created_at ON processing_logs(created_at);
        """
    
    def _get_full_text_search_sql(self) -> str:
        """Get SQL for full-text search setup"""
        return """
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            filename,
            raw_text,
            case_number,
            court_name,
            content='documents',
            content_rowid='id'
        )
        """
    
    def _get_entity_relationships_sql(self) -> str:
        """Get SQL for entity relationship tracking"""
        return """
        -- Add relationship strength/confidence to entity relationships
        ALTER TABLE entity_relationships 
        ADD COLUMN relationship_strength REAL DEFAULT 0.5;
        
        -- Add context for relationships
        ALTER TABLE entity_relationships 
        ADD COLUMN relationship_context TEXT;
        
        -- Create index for relationship queries
        CREATE INDEX IF NOT EXISTS idx_entity_rel_type ON entity_relationships(relationship_type);
        CREATE INDEX IF NOT EXISTS idx_entity_rel_strength ON entity_relationships(relationship_strength);
        """
    
    def create_sample_data(self):
        """Create sample data for testing"""
        try:
            with self.db.get_session() as session:
                # Check if sample data already exists
                existing = session.execute(
                    text("SELECT COUNT(*) FROM documents WHERE filename LIKE 'sample_%'")
                ).scalar()
                
                if existing > 0:
                    logger.info("Sample data already exists, skipping creation")
                    return
                
                # Create sample documents
                sample_sql = """
                INSERT INTO documents (
                    filename, file_path, file_hash, document_type, case_number,
                    court_name, processing_status, processing_time, raw_text,
                    text_length, word_count, confidence_score, created_at
                ) VALUES 
                (
                    'sample_criminal_judgment.pdf',
                    '/sample/criminal.pdf',
                    'sample_hash_1',
                    'CRIMINAL_JUDGMENT',
                    'BA 123/2024/HS.ST',
                    'Tòa án nhân dân TP. Hồ Chí Minh',
                    'COMPLETED',
                    2.5,
                    'Bản án hình sự mẫu về tội trộm cắp tài sản...',
                    1500,
                    250,
                    0.85,
                    datetime('now')
                ),
                (
                    'sample_civil_judgment.pdf',
                    '/sample/civil.pdf',
                    'sample_hash_2',
                    'CIVIL_JUDGMENT',
                    'BA 456/2024/DS.ST',
                    'Tòa án nhân dân quận 1',
                    'COMPLETED',
                    3.2,
                    'Bản án dân sự mẫu về tranh chấp hợp đồng...',
                    2000,
                    350,
                    0.78,
                    datetime('now')
                );
                """
                
                session.execute(text(sample_sql))
                logger.info("Sample data created successfully")
                
        except Exception as e:
            logger.error(f"Error creating sample data: {e}")
            raise
    
    def backup_database(self, backup_path: str):
        """Create database backup"""
        try:
            import shutil
            import os
            
            db_path = self.db.database_url.replace('sqlite:///', '')
            
            if os.path.exists(db_path):
                shutil.copy2(db_path, backup_path)
                logger.info(f"Database backed up to {backup_path}")
            else:
                logger.warning(f"Database file not found: {db_path}")
                
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            raise
    
    def restore_database(self, backup_path: str):
        """Restore database from backup"""
        try:
            import shutil
            import os
            
            if not os.path.exists(backup_path):
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            db_path = self.db.database_url.replace('sqlite:///', '')
            
            # Close existing connections
            self.db.close_all_connections()
            
            # Restore backup
            shutil.copy2(backup_path, db_path)
            logger.info(f"Database restored from {backup_path}")
            
        except Exception as e:
            logger.error(f"Error restoring database: {e}")
            raise
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status"""
        try:
            applied_migrations = self.get_applied_migrations()
            all_migrations = [version for version, _, _ in self.get_migration_list()]
            
            pending_migrations = [
                version for version in all_migrations 
                if version not in applied_migrations
            ]
            
            return {
                'total_migrations': len(all_migrations),
                'applied_migrations': len(applied_migrations),
                'pending_migrations': len(pending_migrations),
                'applied_list': applied_migrations,
                'pending_list': pending_migrations,
                'database_version': applied_migrations[-1] if applied_migrations else None
            }
            
        except Exception as e:
            logger.error(f"Error getting migration status: {e}")
            return {
                'error': str(e),
                'total_migrations': 0,
                'applied_migrations': 0,
                'pending_migrations': 0
            }


def init_database_with_migrations():
    """Initialize database with all migrations"""
    try:
        # Create database connection
        db = get_database_connection()
        
        # Create all tables using SQLAlchemy
        db.create_tables()
        
        # Run migrations
        migration_manager = DatabaseMigration(db)
        migration_manager.run_all_migrations()
        
        logger.info("Database initialized successfully with migrations")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


if __name__ == "__main__":
    # Run migrations when script is executed directly
    logging.basicConfig(level=logging.INFO)
    init_database_with_migrations()
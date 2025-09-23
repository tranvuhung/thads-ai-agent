#!/usr/bin/env python3
"""
Database initialization script for Legal Document Knowledge Base
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import get_database_connection
from database.migrations import DatabaseMigration, init_database_with_migrations
from database.knowledge_base import KnowledgeBase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_database_directory():
    """Create database directory if it doesn't exist"""
    db_dir = Path(__file__).parent.parent.parent / 'data' / 'database'
    db_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Database directory created: {db_dir}")
    return db_dir


def initialize_database(create_sample_data: bool = False, backup_existing: bool = True):
    """
    Initialize the legal document database
    
    Args:
        create_sample_data: Whether to create sample data for testing
        backup_existing: Whether to backup existing database before initialization
    """
    try:
        logger.info("Starting database initialization...")
        
        # Create database directory
        db_dir = create_database_directory()
        
        # Get database connection
        db = get_database_connection()
        
        # Backup existing database if requested
        if backup_existing:
            db_path = db.database_url.replace('sqlite:///', '')
            if os.path.exists(db_path):
                backup_path = f"{db_path}.backup.{int(datetime.now().timestamp())}"
                migration_manager = DatabaseMigration(db)
                migration_manager.backup_database(backup_path)
                logger.info(f"Existing database backed up to: {backup_path}")
        
        # Initialize database with migrations
        init_database_with_migrations()
        
        # Create sample data if requested
        if create_sample_data:
            migration_manager = DatabaseMigration(db)
            migration_manager.create_sample_data()
            logger.info("Sample data created")
        
        # Verify database setup
        verify_database_setup(db)
        
        logger.info("Database initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def verify_database_setup(db):
    """Verify that database is properly set up"""
    try:
        # Test database connection
        info = db.get_database_info()
        logger.info(f"Database info: {info}")
        
        # Test Knowledge Base
        kb = KnowledgeBase(db)
        stats = kb.get_statistics()
        logger.info(f"Knowledge Base statistics: {stats}")
        
        # Test migration status
        migration_manager = DatabaseMigration(db)
        status = migration_manager.get_migration_status()
        logger.info(f"Migration status: {status}")
        
        logger.info("Database verification completed successfully")
        
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        raise


def reset_database():
    """Reset database by removing all data"""
    try:
        logger.warning("Resetting database - all data will be lost!")
        
        db = get_database_connection()
        db_path = db.database_url.replace('sqlite:///', '')
        
        if os.path.exists(db_path):
            # Backup before reset
            backup_path = f"{db_path}.reset_backup.{int(datetime.now().timestamp())}"
            migration_manager = DatabaseMigration(db)
            migration_manager.backup_database(backup_path)
            logger.info(f"Database backed up to: {backup_path}")
            
            # Close connections and remove database file
            db.close_all_connections()
            os.remove(db_path)
            logger.info("Database file removed")
        
        # Reinitialize
        initialize_database(create_sample_data=False, backup_existing=False)
        logger.info("Database reset completed")
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        raise


def show_database_status():
    """Show current database status"""
    try:
        db = get_database_connection()
        
        # Database info
        info = db.get_database_info()
        print(f"\n=== Database Information ===")
        print(f"URL: {info.get('url', 'N/A')}")
        print(f"Size: {info.get('size', 'N/A')}")
        print(f"Tables: {info.get('table_count', 'N/A')}")
        
        # Migration status
        migration_manager = DatabaseMigration(db)
        status = migration_manager.get_migration_status()
        print(f"\n=== Migration Status ===")
        print(f"Total migrations: {status['total_migrations']}")
        print(f"Applied migrations: {status['applied_migrations']}")
        print(f"Pending migrations: {status['pending_migrations']}")
        print(f"Database version: {status.get('database_version', 'N/A')}")
        
        if status['pending_list']:
            print(f"Pending: {', '.join(status['pending_list'])}")
        
        # Knowledge Base statistics
        kb = KnowledgeBase(db)
        stats = kb.get_statistics()
        print(f"\n=== Knowledge Base Statistics ===")
        print(f"Total documents: {stats['documents']['total']}")
        print(f"Total entities: {stats['entities']['total']}")
        print(f"Average confidence: {stats['documents']['avg_confidence_score']:.2f}")
        print(f"Average processing time: {stats['documents']['avg_processing_time']:.2f}s")
        
        if stats['documents']['by_type']:
            print(f"\nDocument types:")
            for doc_type, count in stats['documents']['by_type'].items():
                print(f"  {doc_type}: {count}")
        
        if stats['entities']['by_type']:
            print(f"\nEntity types:")
            for entity_type, count in stats['entities']['by_type'].items():
                print(f"  {entity_type}: {count}")
        
    except Exception as e:
        logger.error(f"Error showing database status: {e}")
        raise


def run_migrations():
    """Run pending database migrations"""
    try:
        logger.info("Running database migrations...")
        
        db = get_database_connection()
        migration_manager = DatabaseMigration(db)
        migration_manager.run_all_migrations()
        
        logger.info("Migrations completed successfully")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Legal Document Knowledge Base Database Management"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize database')
    init_parser.add_argument(
        '--sample-data', action='store_true',
        help='Create sample data for testing'
    )
    init_parser.add_argument(
        '--no-backup', action='store_true',
        help='Skip backing up existing database'
    )
    
    # Status command
    subparsers.add_parser('status', help='Show database status')
    
    # Reset command
    subparsers.add_parser('reset', help='Reset database (removes all data)')
    
    # Migrate command
    subparsers.add_parser('migrate', help='Run pending migrations')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create database backup')
    backup_parser.add_argument('backup_path', help='Backup file path')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore database from backup')
    restore_parser.add_argument('backup_path', help='Backup file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'init':
            initialize_database(
                create_sample_data=args.sample_data,
                backup_existing=not args.no_backup
            )
        
        elif args.command == 'status':
            show_database_status()
        
        elif args.command == 'reset':
            # Confirm reset
            response = input("Are you sure you want to reset the database? (yes/no): ")
            if response.lower() == 'yes':
                reset_database()
            else:
                print("Reset cancelled")
        
        elif args.command == 'migrate':
            run_migrations()
        
        elif args.command == 'backup':
            db = get_database_connection()
            migration_manager = DatabaseMigration(db)
            migration_manager.backup_database(args.backup_path)
            print(f"Database backed up to: {args.backup_path}")
        
        elif args.command == 'restore':
            db = get_database_connection()
            migration_manager = DatabaseMigration(db)
            migration_manager.restore_database(args.backup_path)
            print(f"Database restored from: {args.backup_path}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
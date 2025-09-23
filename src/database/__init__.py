"""
Database package for Legal Document Knowledge Base
"""

from .models import *
from .connection import DatabaseConnection
from .knowledge_base import KnowledgeBase

__all__ = [
    'DatabaseConnection',
    'KnowledgeBase',
    'Document',
    'LegalEntity',
    'Person',
    'Organization',
    'LegalReference',
    'Case',
    'DocumentEntity',
    'EntityRelationship'
]
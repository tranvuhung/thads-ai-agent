"""
Context Management System

This module provides comprehensive context management for the legal AI agent, including:
- Conversation memory and history tracking
- Document context management
- Knowledge base integration
- Session management
- Context summarization and compression
- Relevance scoring and retrieval
"""

import asyncio
import json
import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
import sqlite3
import pickle
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context information"""
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    KNOWLEDGE_BASE = "knowledge_base"
    SESSION = "session"
    USER_PROFILE = "user_profile"
    LEGAL_CASE = "legal_case"
    REGULATION = "regulation"
    PRECEDENT = "precedent"


class MessageRole(Enum):
    """Roles in conversation messages"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ContextPriority(Enum):
    """Priority levels for context information"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ConversationMessage:
    """Individual message in a conversation"""
    role: MessageRole
    content: str
    timestamp: datetime
    message_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8])
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "metadata": self.metadata,
            "tokens_used": self.tokens_used,
            "response_time": self.response_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        """Create message from dictionary"""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_id=data.get("message_id", ""),
            metadata=data.get("metadata", {}),
            tokens_used=data.get("tokens_used"),
            response_time=data.get("response_time")
        )


@dataclass
class DocumentContext:
    """Context information for a document"""
    document_id: str
    title: str
    content: str
    document_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None
    key_points: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document context to dictionary"""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "content": self.content,
            "document_type": self.document_type,
            "metadata": self.metadata,
            "summary": self.summary,
            "key_points": self.key_points,
            "relevance_score": self.relevance_score,
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentContext':
        """Create document context from dictionary"""
        return cls(
            document_id=data["document_id"],
            title=data["title"],
            content=data["content"],
            document_type=data["document_type"],
            metadata=data.get("metadata", {}),
            summary=data.get("summary"),
            key_points=data.get("key_points", []),
            relevance_score=data.get("relevance_score", 0.0),
            last_accessed=datetime.fromisoformat(data.get("last_accessed", datetime.now().isoformat())),
            access_count=data.get("access_count", 0)
        )


@dataclass
class ContextEntry:
    """Generic context entry"""
    context_id: str
    context_type: ContextType
    content: Any
    priority: ContextPriority
    timestamp: datetime
    expiry: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0
    access_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if context entry has expired"""
        if self.expiry is None:
            return False
        return datetime.now() > self.expiry
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context entry to dictionary"""
        return {
            "context_id": self.context_id,
            "context_type": self.context_type.value,
            "content": self.content,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "metadata": self.metadata,
            "relevance_score": self.relevance_score,
            "access_count": self.access_count
        }


@dataclass
class ConversationSession:
    """Conversation session with context"""
    session_id: str
    user_id: Optional[str] = None
    messages: List[ConversationMessage] = field(default_factory=list)
    document_contexts: List[DocumentContext] = field(default_factory=list)
    context_entries: List[ContextEntry] = field(default_factory=list)
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    total_tokens_used: int = 0
    
    def add_message(self, message: ConversationMessage):
        """Add a message to the session"""
        self.messages.append(message)
        self.last_activity = datetime.now()
        if message.tokens_used:
            self.total_tokens_used += message.tokens_used
    
    def add_document_context(self, document: DocumentContext):
        """Add document context to the session"""
        # Remove existing document with same ID
        self.document_contexts = [doc for doc in self.document_contexts 
                                 if doc.document_id != document.document_id]
        self.document_contexts.append(document)
        self.last_activity = datetime.now()
    
    def add_context_entry(self, context: ContextEntry):
        """Add context entry to the session"""
        self.context_entries.append(context)
        self.last_activity = datetime.now()
    
    def get_recent_messages(self, limit: int = 10) -> List[ConversationMessage]:
        """Get recent messages from the session"""
        return self.messages[-limit:] if self.messages else []
    
    def get_conversation_summary(self, max_length: int = 500) -> str:
        """Get a summary of the conversation"""
        if not self.messages:
            return "No conversation history"
        
        # Simple summary - take first and last few messages
        summary_messages = []
        if len(self.messages) <= 4:
            summary_messages = self.messages
        else:
            summary_messages = self.messages[:2] + self.messages[-2:]
        
        summary = []
        for msg in summary_messages:
            role = msg.role.value.upper()
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary.append(f"{role}: {content}")
        
        full_summary = "\n".join(summary)
        if len(full_summary) > max_length:
            return full_summary[:max_length] + "..."
        return full_summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "document_contexts": [doc.to_dict() for doc in self.document_contexts],
            "context_entries": [ctx.to_dict() for ctx in self.context_entries],
            "session_metadata": self.session_metadata,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "total_tokens_used": self.total_tokens_used
        }


class ContextStorage(ABC):
    """Abstract base class for context storage"""
    
    @abstractmethod
    async def save_session(self, session: ConversationSession) -> bool:
        """Save conversation session"""
        pass
    
    @abstractmethod
    async def load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load conversation session"""
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete conversation session"""
        pass
    
    @abstractmethod
    async def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """List session IDs"""
        pass


class SQLiteContextStorage(ContextStorage):
    """SQLite-based context storage implementation"""
    
    def __init__(self, db_path: str = "context_storage.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                session_data BLOB,
                created_at TEXT,
                last_activity TEXT,
                total_tokens_used INTEGER
            )
        """)
        
        # Create context entries table for quick queries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS context_entries (
                context_id TEXT PRIMARY KEY,
                session_id TEXT,
                context_type TEXT,
                priority TEXT,
                timestamp TEXT,
                expiry TEXT,
                relevance_score REAL,
                access_count INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def save_session(self, session: ConversationSession) -> bool:
        """Save conversation session to SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Serialize session data
            session_data = pickle.dumps(session.to_dict())
            
            cursor.execute("""
                INSERT OR REPLACE INTO sessions 
                (session_id, user_id, session_data, created_at, last_activity, total_tokens_used)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.user_id,
                session_data,
                session.created_at.isoformat(),
                session.last_activity.isoformat(),
                session.total_tokens_used
            ))
            
            # Save context entries for quick queries
            cursor.execute("DELETE FROM context_entries WHERE session_id = ?", (session.session_id,))
            
            for context in session.context_entries:
                cursor.execute("""
                    INSERT INTO context_entries 
                    (context_id, session_id, context_type, priority, timestamp, expiry, relevance_score, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    context.context_id,
                    session.session_id,
                    context.context_type.value,
                    context.priority.value,
                    context.timestamp.isoformat(),
                    context.expiry.isoformat() if context.expiry else None,
                    context.relevance_score,
                    context.access_count
                ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {str(e)}")
            return False
    
    async def load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load conversation session from SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT session_data FROM sessions WHERE session_id = ?", (session_id,))
            result = cursor.fetchone()
            
            if result:
                session_data = pickle.loads(result[0])
                
                # Reconstruct session object
                session = ConversationSession(
                    session_id=session_data["session_id"],
                    user_id=session_data.get("user_id"),
                    session_metadata=session_data.get("session_metadata", {}),
                    created_at=datetime.fromisoformat(session_data["created_at"]),
                    last_activity=datetime.fromisoformat(session_data["last_activity"]),
                    total_tokens_used=session_data.get("total_tokens_used", 0)
                )
                
                # Reconstruct messages
                for msg_data in session_data.get("messages", []):
                    session.messages.append(ConversationMessage.from_dict(msg_data))
                
                # Reconstruct document contexts
                for doc_data in session_data.get("document_contexts", []):
                    session.document_contexts.append(DocumentContext.from_dict(doc_data))
                
                # Reconstruct context entries
                for ctx_data in session_data.get("context_entries", []):
                    context = ContextEntry(
                        context_id=ctx_data["context_id"],
                        context_type=ContextType(ctx_data["context_type"]),
                        content=ctx_data["content"],
                        priority=ContextPriority(ctx_data["priority"]),
                        timestamp=datetime.fromisoformat(ctx_data["timestamp"]),
                        expiry=datetime.fromisoformat(ctx_data["expiry"]) if ctx_data.get("expiry") else None,
                        metadata=ctx_data.get("metadata", {}),
                        relevance_score=ctx_data.get("relevance_score", 0.0),
                        access_count=ctx_data.get("access_count", 0)
                    )
                    session.context_entries.append(context)
                
                conn.close()
                return session
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {str(e)}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete conversation session from SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM context_entries WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {str(e)}")
            return False
    
    async def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """List session IDs from SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute("SELECT session_id FROM sessions WHERE user_id = ? ORDER BY last_activity DESC", (user_id,))
            else:
                cursor.execute("SELECT session_id FROM sessions ORDER BY last_activity DESC")
            
            results = cursor.fetchall()
            conn.close()
            
            return [row[0] for row in results]
            
        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}")
            return []


class ContextManager:
    """Main context management system"""
    
    def __init__(self, storage: Optional[ContextStorage] = None):
        self.storage = storage or SQLiteContextStorage()
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.context_cache: Dict[str, Any] = {}
        self.max_context_length = 8000  # Maximum context length in tokens
        self.max_messages_in_context = 20
    
    async def create_session(self, session_id: str, user_id: Optional[str] = None) -> ConversationSession:
        """Create a new conversation session"""
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id
        )
        
        self.active_sessions[session_id] = session
        await self.storage.save_session(session)
        
        logger.info(f"Created new session: {session_id}")
        return session
    
    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get conversation session"""
        # Check active sessions first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Load from storage
        session = await self.storage.load_session(session_id)
        if session:
            self.active_sessions[session_id] = session
        
        return session
    
    async def add_message(
        self, 
        session_id: str, 
        role: MessageRole, 
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tokens_used: Optional[int] = None,
        response_time: Optional[float] = None
    ) -> bool:
        """Add a message to the conversation"""
        session = await self.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return False
        
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
            tokens_used=tokens_used,
            response_time=response_time
        )
        
        session.add_message(message)
        await self.storage.save_session(session)
        
        logger.debug(f"Added message to session {session_id}: {role.value}")
        return True
    
    async def add_document_context(
        self, 
        session_id: str, 
        document: DocumentContext
    ) -> bool:
        """Add document context to the session"""
        session = await self.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return False
        
        document.last_accessed = datetime.now()
        document.access_count += 1
        
        session.add_document_context(document)
        await self.storage.save_session(session)
        
        logger.debug(f"Added document context to session {session_id}: {document.document_id}")
        return True
    
    async def add_context_entry(
        self, 
        session_id: str, 
        context: ContextEntry
    ) -> bool:
        """Add context entry to the session"""
        session = await self.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return False
        
        session.add_context_entry(context)
        await self.storage.save_session(session)
        
        logger.debug(f"Added context entry to session {session_id}: {context.context_id}")
        return True
    
    async def get_conversation_context(
        self, 
        session_id: str,
        max_messages: Optional[int] = None,
        include_system: bool = True
    ) -> List[Dict[str, str]]:
        """Get conversation context for LLM"""
        session = await self.get_session(session_id)
        if not session:
            return []
        
        max_messages = max_messages or self.max_messages_in_context
        recent_messages = session.get_recent_messages(max_messages)
        
        context = []
        for message in recent_messages:
            if not include_system and message.role == MessageRole.SYSTEM:
                continue
            
            context.append({
                "role": message.role.value,
                "content": message.content
            })
        
        return context
    
    async def get_relevant_documents(
        self, 
        session_id: str,
        query: Optional[str] = None,
        limit: int = 5
    ) -> List[DocumentContext]:
        """Get relevant documents for the current context"""
        session = await self.get_session(session_id)
        if not session:
            return []
        
        # Simple relevance scoring based on access count and recency
        documents = session.document_contexts.copy()
        
        for doc in documents:
            # Base score from access count
            doc.relevance_score = doc.access_count * 0.1
            
            # Recency bonus (more recent = higher score)
            hours_since_access = (datetime.now() - doc.last_accessed).total_seconds() / 3600
            recency_score = max(0, 1 - (hours_since_access / 24))  # Decay over 24 hours
            doc.relevance_score += recency_score
            
            # Query relevance (simple keyword matching)
            if query:
                query_lower = query.lower()
                title_matches = query_lower in doc.title.lower()
                content_matches = query_lower in doc.content.lower()
                
                if title_matches:
                    doc.relevance_score += 2.0
                if content_matches:
                    doc.relevance_score += 1.0
        
        # Sort by relevance and return top results
        documents.sort(key=lambda x: x.relevance_score, reverse=True)
        return documents[:limit]
    
    async def compress_context(self, session_id: str) -> bool:
        """Compress context to stay within limits"""
        session = await self.get_session(session_id)
        if not session:
            return False
        
        # Remove expired context entries
        session.context_entries = [ctx for ctx in session.context_entries if not ctx.is_expired()]
        
        # If we have too many messages, summarize older ones
        if len(session.messages) > self.max_messages_in_context * 2:
            # Keep recent messages and create summary of older ones
            recent_messages = session.messages[-self.max_messages_in_context:]
            older_messages = session.messages[:-self.max_messages_in_context]
            
            # Create summary message
            summary_content = f"[Previous conversation summary: {len(older_messages)} messages exchanged]"
            summary_message = ConversationMessage(
                role=MessageRole.SYSTEM,
                content=summary_content,
                timestamp=older_messages[0].timestamp if older_messages else datetime.now()
            )
            
            session.messages = [summary_message] + recent_messages
        
        await self.storage.save_session(session)
        logger.info(f"Compressed context for session {session_id}")
        return True
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of session context"""
        session = await self.get_session(session_id)
        if not session:
            return {}
        
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "message_count": len(session.messages),
            "document_count": len(session.document_contexts),
            "context_entries": len(session.context_entries),
            "total_tokens_used": session.total_tokens_used,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "conversation_summary": session.get_conversation_summary()
        }
    
    async def cleanup_expired_sessions(self, max_age_days: int = 30) -> int:
        """Clean up expired sessions"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        session_ids = await self.storage.list_sessions()
        
        cleaned_count = 0
        for session_id in session_ids:
            session = await self.storage.load_session(session_id)
            if session and session.last_activity < cutoff_date:
                await self.storage.delete_session(session_id)
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} expired sessions")
        return cleaned_count


# Factory functions
def create_context_manager(storage_path: Optional[str] = None) -> ContextManager:
    """Create a context manager with SQLite storage"""
    storage = SQLiteContextStorage(storage_path) if storage_path else SQLiteContextStorage()
    return ContextManager(storage)


def create_document_context(
    document_id: str,
    title: str,
    content: str,
    document_type: str,
    summary: Optional[str] = None,
    **metadata
) -> DocumentContext:
    """Create a document context entry"""
    return DocumentContext(
        document_id=document_id,
        title=title,
        content=content,
        document_type=document_type,
        summary=summary,
        metadata=metadata
    )


# Example usage
async def example_usage():
    """Example of how to use the context management system"""
    # Create context manager
    context_manager = create_context_manager()
    
    # Create a new session
    session = await context_manager.create_session("session_123", "user_456")
    
    # Add messages to conversation
    await context_manager.add_message(
        "session_123", 
        MessageRole.USER, 
        "I need help analyzing a contract"
    )
    
    await context_manager.add_message(
        "session_123", 
        MessageRole.ASSISTANT, 
        "I'd be happy to help you analyze the contract. Please share the document."
    )
    
    # Add document context
    doc_context = create_document_context(
        document_id="contract_001",
        title="Software License Agreement",
        content="This agreement governs the use of software...",
        document_type="contract",
        summary="Software licensing terms and conditions"
    )
    
    await context_manager.add_document_context("session_123", doc_context)
    
    # Get conversation context for LLM
    conversation_context = await context_manager.get_conversation_context("session_123")
    print("Conversation context:", conversation_context)
    
    # Get relevant documents
    relevant_docs = await context_manager.get_relevant_documents("session_123", "software license")
    print("Relevant documents:", [doc.title for doc in relevant_docs])
    
    # Get session summary
    summary = await context_manager.get_session_summary("session_123")
    print("Session summary:", summary)


if __name__ == "__main__":
    asyncio.run(example_usage())
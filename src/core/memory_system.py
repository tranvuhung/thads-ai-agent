"""
Memory System for LLM Integration

This module provides an advanced memory system for conversation history and knowledge retention.
It includes sophisticated storage, retrieval, and learning capabilities to enhance AI interactions
with persistent memory, knowledge graphs, and intelligent recall mechanisms.

Key Features:
- Persistent conversation memory with semantic indexing
- Knowledge graph construction and maintenance
- Intelligent memory retrieval and ranking
- Memory consolidation and summarization
- Long-term and short-term memory management
- Contextual memory associations
- Memory decay and importance scoring
- Cross-session knowledge transfer
"""

import asyncio
import json
import logging
import sqlite3
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory entries"""
    CONVERSATION = "conversation"
    KNOWLEDGE = "knowledge"
    FACT = "fact"
    PROCEDURE = "procedure"
    EXPERIENCE = "experience"
    PREFERENCE = "preference"
    CONTEXT = "context"
    SUMMARY = "summary"


class MemoryImportance(Enum):
    """Importance levels for memory entries"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1


class MemoryStatus(Enum):
    """Status of memory entries"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    CONSOLIDATED = "consolidated"
    EXPIRED = "expired"


@dataclass
class MemoryEntry:
    """Individual memory entry with metadata"""
    id: str
    content: str
    memory_type: MemoryType
    importance: MemoryImportance
    timestamp: datetime
    session_id: str
    user_id: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    associations: Set[str] = field(default_factory=set)  # IDs of related memories
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    decay_factor: float = 1.0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: MemoryStatus = MemoryStatus.ACTIVE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory entry to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance.value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tags": list(self.tags),
            "associations": list(self.associations),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "decay_factor": self.decay_factor,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "status": self.status.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create memory entry from dictionary"""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            importance=MemoryImportance(data["importance"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            tags=set(data.get("tags", [])),
            associations=set(data.get("associations", [])),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            decay_factor=data.get("decay_factor", 1.0),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            status=MemoryStatus(data.get("status", "active"))
        )


@dataclass
class MemoryQuery:
    """Query for memory retrieval"""
    query_text: str
    memory_types: Optional[List[MemoryType]] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    tags: Optional[Set[str]] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    importance_threshold: Optional[MemoryImportance] = None
    limit: int = 10
    include_associations: bool = True
    semantic_search: bool = True


@dataclass
class MemorySearchResult:
    """Result of memory search"""
    memory: MemoryEntry
    relevance_score: float
    match_type: str  # "semantic", "keyword", "tag", "association"
    explanation: str


class MemoryEmbedding:
    """Simple embedding system for semantic search"""
    
    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf_scores: Dict[str, float] = {}
        self.dimension = 100
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _build_vocabulary(self, texts: List[str]):
        """Build vocabulary from texts"""
        word_counts = defaultdict(int)
        doc_counts = defaultdict(int)
        
        for text in texts:
            tokens = self._tokenize(text)
            unique_tokens = set(tokens)
            
            for token in tokens:
                word_counts[token] += 1
            
            for token in unique_tokens:
                doc_counts[token] += 1
        
        # Build vocabulary with most frequent words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words[:self.dimension])}
        
        # Calculate IDF scores
        total_docs = len(texts)
        for word in self.vocabulary:
            self.idf_scores[word] = np.log(total_docs / (doc_counts[word] + 1))
    
    def encode(self, text: str) -> List[float]:
        """Encode text to embedding vector"""
        if not self.vocabulary:
            return [0.0] * self.dimension
        
        tokens = self._tokenize(text)
        vector = np.zeros(self.dimension)
        
        for token in tokens:
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                vector[idx] += self.idf_scores.get(token, 1.0)
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        if not embedding1 or not embedding2:
            return 0.0
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class MemoryStorage:
    """Storage backend for memory system"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "memory.db"
        self._init_database()
        self._lock = threading.Lock()
    
    def _init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    user_id TEXT,
                    tags TEXT,
                    associations TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    decay_factor REAL DEFAULT 1.0,
                    embedding BLOB,
                    metadata TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id ON memories(session_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON memories(user_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)
            """)
            
            conn.commit()
    
    async def store_memory(self, memory: MemoryEntry) -> bool:
        """Store a memory entry"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO memories (
                            id, content, memory_type, importance, timestamp,
                            session_id, user_id, tags, associations, access_count,
                            last_accessed, decay_factor, embedding, metadata, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        memory.id,
                        memory.content,
                        memory.memory_type.value,
                        memory.importance.value,
                        memory.timestamp.isoformat(),
                        memory.session_id,
                        memory.user_id,
                        json.dumps(list(memory.tags)),
                        json.dumps(list(memory.associations)),
                        memory.access_count,
                        memory.last_accessed.isoformat() if memory.last_accessed else None,
                        memory.decay_factor,
                        pickle.dumps(memory.embedding) if memory.embedding else None,
                        json.dumps(memory.metadata),
                        memory.status.value
                    ))
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            return False
    
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by ID"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT * FROM memories WHERE id = ?
                    """, (memory_id,))
                    row = cursor.fetchone()
                    
                    if row:
                        return self._row_to_memory(row)
            return None
        except Exception as e:
            logger.error(f"Error retrieving memory: {str(e)}")
            return None
    
    async def search_memories(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Search memories based on query criteria"""
        try:
            conditions = ["status = 'active'"]
            params = []
            
            if query.memory_types:
                type_placeholders = ",".join("?" * len(query.memory_types))
                conditions.append(f"memory_type IN ({type_placeholders})")
                params.extend([t.value for t in query.memory_types])
            
            if query.session_id:
                conditions.append("session_id = ?")
                params.append(query.session_id)
            
            if query.user_id:
                conditions.append("user_id = ?")
                params.append(query.user_id)
            
            if query.importance_threshold:
                conditions.append("importance >= ?")
                params.append(query.importance_threshold.value)
            
            if query.time_range:
                conditions.append("timestamp BETWEEN ? AND ?")
                params.extend([t.isoformat() for t in query.time_range])
            
            # Text search
            if query.query_text:
                conditions.append("content LIKE ?")
                params.append(f"%{query.query_text}%")
            
            where_clause = " AND ".join(conditions)
            sql = f"""
                SELECT * FROM memories 
                WHERE {where_clause}
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
            """
            params.append(query.limit)
            
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(sql, params)
                    rows = cursor.fetchall()
                    
                    memories = [self._row_to_memory(row) for row in rows]
                    
                    # Filter by tags if specified
                    if query.tags:
                        memories = [
                            m for m in memories 
                            if query.tags.intersection(m.tags)
                        ]
                    
                    return memories[:query.limit]
        except Exception as e:
            logger.error(f"Error searching memories: {str(e)}")
            return []
    
    async def update_memory_access(self, memory_id: str) -> bool:
        """Update memory access statistics"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE memories 
                        SET access_count = access_count + 1,
                            last_accessed = ?
                        WHERE id = ?
                    """, (datetime.now().isoformat(), memory_id))
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating memory access: {str(e)}")
            return False
    
    async def get_all_memories(self) -> List[MemoryEntry]:
        """Get all memories for embedding training"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT * FROM memories WHERE status = 'active'")
                    rows = cursor.fetchall()
                    return [self._row_to_memory(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting all memories: {str(e)}")
            return []
    
    def _row_to_memory(self, row) -> MemoryEntry:
        """Convert database row to MemoryEntry"""
        return MemoryEntry(
            id=row[0],
            content=row[1],
            memory_type=MemoryType(row[2]),
            importance=MemoryImportance(row[3]),
            timestamp=datetime.fromisoformat(row[4]),
            session_id=row[5],
            user_id=row[6],
            tags=set(json.loads(row[7]) if row[7] else []),
            associations=set(json.loads(row[8]) if row[8] else []),
            access_count=row[9] or 0,
            last_accessed=datetime.fromisoformat(row[10]) if row[10] else None,
            decay_factor=row[11] or 1.0,
            embedding=pickle.loads(row[12]) if row[12] else None,
            metadata=json.loads(row[13]) if row[13] else {},
            status=MemoryStatus(row[14] or "active")
        )


class MemoryConsolidator:
    """Consolidates and summarizes memories"""
    
    def __init__(self, storage: MemoryStorage):
        self.storage = storage
    
    async def consolidate_session_memories(self, session_id: str) -> Optional[MemoryEntry]:
        """Consolidate memories from a session into a summary"""
        query = MemoryQuery(
            query_text="",
            session_id=session_id,
            memory_types=[MemoryType.CONVERSATION],
            limit=100
        )
        
        memories = await self.storage.search_memories(query)
        
        if len(memories) < 3:  # Not enough to consolidate
            return None
        
        # Create summary content
        summary_content = self._create_session_summary(memories)
        
        # Create consolidated memory
        consolidated_memory = MemoryEntry(
            id=f"summary_{session_id}_{datetime.now().timestamp()}",
            content=summary_content,
            memory_type=MemoryType.SUMMARY,
            importance=MemoryImportance.MEDIUM,
            timestamp=datetime.now(),
            session_id=session_id,
            user_id=memories[0].user_id if memories else None,
            tags={"session_summary", "consolidated"},
            metadata={
                "original_memory_count": len(memories),
                "consolidation_date": datetime.now().isoformat(),
                "session_id": session_id
            }
        )
        
        # Store consolidated memory
        await self.storage.store_memory(consolidated_memory)
        
        # Mark original memories as consolidated
        for memory in memories:
            memory.status = MemoryStatus.CONSOLIDATED
            await self.storage.store_memory(memory)
        
        return consolidated_memory
    
    def _create_session_summary(self, memories: List[MemoryEntry]) -> str:
        """Create a summary from session memories"""
        if not memories:
            return "Empty session"
        
        # Sort by timestamp
        sorted_memories = sorted(memories, key=lambda m: m.timestamp)
        
        # Extract key topics and themes
        topics = set()
        for memory in sorted_memories:
            topics.update(memory.tags)
        
        # Create summary
        summary_parts = [
            f"Session Summary ({len(memories)} interactions)",
            f"Duration: {sorted_memories[0].timestamp.strftime('%Y-%m-%d %H:%M')} - {sorted_memories[-1].timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"Key topics: {', '.join(list(topics)[:5])}" if topics else "No specific topics identified",
            "",
            "Key interactions:"
        ]
        
        # Add important interactions
        important_memories = [m for m in memories if m.importance.value >= 3]
        for memory in important_memories[:5]:
            summary_parts.append(f"- {memory.content[:100]}...")
        
        return "\n".join(summary_parts)


class MemorySystem:
    """Main memory system that orchestrates all components"""
    
    def __init__(self, storage_path: str = "data/memory"):
        self.storage = MemoryStorage(storage_path)
        self.embedding_system = MemoryEmbedding()
        self.consolidator = MemoryConsolidator(self.storage)
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Memory management settings
        self.max_memories_per_session = 1000
        self.consolidation_threshold = 50
        self.decay_rate = 0.01  # Daily decay rate
        
        # Initialize embedding system
        asyncio.create_task(self._initialize_embeddings())
    
    async def _initialize_embeddings(self):
        """Initialize embedding system with existing memories"""
        try:
            memories = await self.storage.get_all_memories()
            if memories:
                texts = [memory.content for memory in memories]
                self.embedding_system._build_vocabulary(texts)
                
                # Update embeddings for existing memories
                for memory in memories:
                    if not memory.embedding:
                        memory.embedding = self.embedding_system.encode(memory.content)
                        await self.storage.store_memory(memory)
                
                logger.info(f"Initialized embeddings for {len(memories)} memories")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
    
    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        session_id: str,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        user_id: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a new memory"""
        # Generate unique ID
        memory_id = hashlib.md5(
            f"{content}_{session_id}_{datetime.now().timestamp()}".encode()
        ).hexdigest()
        
        # Generate embedding
        embedding = self.embedding_system.encode(content)
        
        # Create memory entry
        memory = MemoryEntry(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            timestamp=datetime.now(),
            session_id=session_id,
            user_id=user_id,
            tags=tags or set(),
            embedding=embedding,
            metadata=metadata or {}
        )
        
        # Store memory
        success = await self.storage.store_memory(memory)
        
        if success:
            # Check if consolidation is needed
            await self._check_consolidation_needed(session_id)
            
            return memory_id
        else:
            raise Exception("Failed to store memory")
    
    async def retrieve_memories(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        semantic_search: bool = True
    ) -> List[MemorySearchResult]:
        """Retrieve relevant memories"""
        # Create query
        query = MemoryQuery(
            query_text=query_text,
            session_id=session_id,
            user_id=user_id,
            memory_types=memory_types,
            limit=limit * 2,  # Get more for ranking
            semantic_search=semantic_search
        )
        
        # Search memories
        memories = await self.storage.search_memories(query)
        
        # Rank memories
        results = await self._rank_memories(query_text, memories, semantic_search)
        
        # Update access statistics
        for result in results[:limit]:
            await self.storage.update_memory_access(result.memory.id)
        
        return results[:limit]
    
    async def _rank_memories(
        self,
        query_text: str,
        memories: List[MemoryEntry],
        semantic_search: bool
    ) -> List[MemorySearchResult]:
        """Rank memories by relevance"""
        if not memories:
            return []
        
        results = []
        query_embedding = self.embedding_system.encode(query_text) if semantic_search else None
        
        for memory in memories:
            relevance_score = 0.0
            match_type = "keyword"
            explanation = "Keyword match"
            
            # Keyword matching
            if query_text.lower() in memory.content.lower():
                relevance_score += 0.5
                match_type = "keyword"
                explanation = "Direct keyword match"
            
            # Tag matching
            query_words = set(query_text.lower().split())
            if query_words.intersection(memory.tags):
                relevance_score += 0.3
                match_type = "tag"
                explanation = "Tag match"
            
            # Semantic similarity
            if semantic_search and query_embedding and memory.embedding:
                semantic_score = self.embedding_system.similarity(
                    query_embedding, memory.embedding
                )
                if semantic_score > relevance_score:
                    relevance_score = semantic_score
                    match_type = "semantic"
                    explanation = f"Semantic similarity: {semantic_score:.2f}"
            
            # Apply importance and recency boost
            importance_boost = memory.importance.value / 5.0
            recency_boost = self._calculate_recency_boost(memory.timestamp)
            access_boost = min(memory.access_count / 10.0, 0.2)
            
            final_score = relevance_score + importance_boost + recency_boost + access_boost
            
            results.append(MemorySearchResult(
                memory=memory,
                relevance_score=final_score,
                match_type=match_type,
                explanation=explanation
            ))
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results
    
    def _calculate_recency_boost(self, timestamp: datetime) -> float:
        """Calculate recency boost for memory ranking"""
        days_old = (datetime.now() - timestamp).days
        return max(0, 0.2 - (days_old * 0.01))
    
    async def add_memory_association(self, memory_id1: str, memory_id2: str) -> bool:
        """Add association between two memories"""
        memory1 = await self.storage.retrieve_memory(memory_id1)
        memory2 = await self.storage.retrieve_memory(memory_id2)
        
        if memory1 and memory2:
            memory1.associations.add(memory_id2)
            memory2.associations.add(memory_id1)
            
            success1 = await self.storage.store_memory(memory1)
            success2 = await self.storage.store_memory(memory2)
            
            return success1 and success2
        
        return False
    
    async def get_memory_associations(self, memory_id: str) -> List[MemoryEntry]:
        """Get associated memories"""
        memory = await self.storage.retrieve_memory(memory_id)
        if not memory:
            return []
        
        associated_memories = []
        for assoc_id in memory.associations:
            assoc_memory = await self.storage.retrieve_memory(assoc_id)
            if assoc_memory:
                associated_memories.append(assoc_memory)
        
        return associated_memories
    
    async def _check_consolidation_needed(self, session_id: str):
        """Check if session needs memory consolidation"""
        query = MemoryQuery(
            query_text="",
            session_id=session_id,
            memory_types=[MemoryType.CONVERSATION],
            limit=self.consolidation_threshold + 1
        )
        
        memories = await self.storage.search_memories(query)
        
        if len(memories) >= self.consolidation_threshold:
            # Schedule consolidation in background
            asyncio.create_task(self.consolidator.consolidate_session_memories(session_id))
    
    async def cleanup_expired_memories(self, days_old: int = 30):
        """Clean up old, low-importance memories"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        query = MemoryQuery(
            query_text="",
            time_range=(datetime.min, cutoff_date),
            importance_threshold=MemoryImportance.LOW,
            limit=1000
        )
        
        old_memories = await self.storage.search_memories(query)
        
        # Mark low-importance, rarely accessed memories as expired
        for memory in old_memories:
            if (memory.importance == MemoryImportance.LOW and 
                memory.access_count < 2):
                memory.status = MemoryStatus.EXPIRED
                await self.storage.store_memory(memory)
        
        logger.info(f"Cleaned up {len(old_memories)} old memories")
    
    async def get_session_summary(self, session_id: str) -> Optional[str]:
        """Get summary of a session"""
        # Look for existing summary
        query = MemoryQuery(
            query_text="",
            session_id=session_id,
            memory_types=[MemoryType.SUMMARY],
            limit=1
        )
        
        summaries = await self.storage.search_memories(query)
        if summaries:
            return summaries[0].content
        
        # Create new summary if needed
        summary_memory = await self.consolidator.consolidate_session_memories(session_id)
        return summary_memory.content if summary_memory else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        # This would be implemented with actual database queries
        return {
            "total_memories": 0,
            "active_memories": 0,
            "consolidated_memories": 0,
            "memory_types": {},
            "average_importance": 0.0,
            "most_accessed_memories": []
        }


# Factory function
def create_memory_system(storage_path: str = "data/memory") -> MemorySystem:
    """Create a memory system with default configuration"""
    return MemorySystem(storage_path)


# Example usage
async def example_usage():
    """Example of how to use the memory system"""
    # Create memory system
    memory_system = create_memory_system()
    
    # Store some memories
    memory_id1 = await memory_system.store_memory(
        content="User asked about employment contract analysis",
        memory_type=MemoryType.CONVERSATION,
        session_id="session_001",
        importance=MemoryImportance.MEDIUM,
        user_id="user_123",
        tags={"employment", "contract", "analysis"}
    )
    
    memory_id2 = await memory_system.store_memory(
        content="Vietnamese labor law requires specific termination procedures",
        memory_type=MemoryType.KNOWLEDGE,
        session_id="session_001",
        importance=MemoryImportance.HIGH,
        tags={"vietnamese_law", "termination", "procedures"}
    )
    
    # Create association between memories
    await memory_system.add_memory_association(memory_id1, memory_id2)
    
    # Retrieve relevant memories
    results = await memory_system.retrieve_memories(
        query_text="employment contract termination",
        session_id="session_001",
        limit=5
    )
    
    print("Retrieved memories:")
    for result in results:
        print(f"- {result.memory.content[:100]}... (Score: {result.relevance_score:.2f})")
    
    # Get session summary
    summary = await memory_system.get_session_summary("session_001")
    print(f"\nSession summary: {summary}")


if __name__ == "__main__":
    asyncio.run(example_usage())
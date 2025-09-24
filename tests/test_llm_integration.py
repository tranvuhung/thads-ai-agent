"""
Comprehensive Test Suite for LLM Integration

This module provides extensive testing for all LLM integration components including:
- LLM Integration Core (llm_integration.py)
- Prompt Templates (prompt_templates.py)
- Context Management (context_management.py)
- Legal Prompt Engine (legal_prompt_engine.py)
- Memory System (memory_system.py)

Test Categories:
- Unit Tests: Individual component testing
- Integration Tests: Component interaction testing
- Performance Tests: Load and response time testing
- End-to-End Tests: Complete workflow testing
"""

import asyncio
import json
import os
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sqlite3

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.llm_integration import (
    LLMManager, ClaudeClient, LLMConfig, LLMRequest, LLMResponse,
    ModelType, LLMProvider, ResponseFormat, RateLimiter,
    create_claude_config, create_llm_manager
)
from core.prompt_templates import (
    LegalPromptTemplates, PromptTemplateManager, PromptTemplate,
    LegalTaskType, LegalJurisdiction, DocumentType,
    get_legal_prompt, create_prompt_manager
)
from core.context_management import (
    ContextManager, ConversationSession, DocumentContext, ContextEntry,
    ContextType, MessageRole, ContextPriority, SQLiteContextStorage,
    create_context_manager, create_document_context
)
from core.legal_prompt_engine import (
    LegalPromptEngine, LegalQuery, PromptStrategy, ResponseMode,
    TaskAnalyzer, ContextEnricher, LegalResponse,
    create_legal_prompt_engine
)
from core.memory_system import (
    MemorySystem, MemoryEntry, MemoryType, MemoryImportance, MemoryStatus,
    MemoryQuery, MemorySearchResult, MemoryStorage, MemoryEmbedding,
    create_memory_system
)


class TestLLMIntegration:
    """Test suite for LLM Integration core functionality"""
    
    def test_llm_config_creation(self):
        """Test LLM configuration creation"""
        config = create_claude_config()
        
        assert config.provider == LLMProvider.CLAUDE
        assert config.model == ModelType.CLAUDE_3_5_SONNET
        assert config.max_tokens > 0
        assert config.temperature >= 0 and config.temperature <= 1
    
    def test_llm_request_creation(self):
        """Test LLM request structure"""
        messages = [{"role": "user", "content": "Test message"}]
        request = LLMRequest(
            messages=messages,
            system_prompt="Test system prompt",
            max_tokens=1000
        )
        
        assert request.messages == messages
        assert request.system_prompt == "Test system prompt"
        assert request.max_tokens == 1000
        assert request.response_format == ResponseFormat.TEXT
    
    def test_rate_limiter(self):
        """Test rate limiting functionality"""
        limiter = RateLimiter(requests_per_minute=60)
        
        # Should allow first request
        assert limiter.can_make_request()
        limiter.record_request()
        
        # Test rate limiting logic
        assert len(limiter.request_times) == 1
    
    @patch('anthropic.AsyncAnthropic')
    async def test_claude_client_initialization(self, mock_anthropic):
        """Test Claude client initialization"""
        config = create_claude_config()
        client = ClaudeClient(config)
        
        assert client.config == config
        assert client.rate_limiter is not None
    
    @patch('anthropic.AsyncAnthropic')
    async def test_claude_client_generate(self, mock_anthropic):
        """Test Claude client response generation"""
        # Mock the Anthropic client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)
        mock_response.stop_reason = "end_turn"
        
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.return_value = mock_client
        
        config = create_claude_config()
        client = ClaudeClient(config)
        
        request = LLMRequest(
            messages=[{"role": "user", "content": "Test"}],
            system_prompt="Test system"
        )
        
        response = await client.generate(request)
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.usage["input_tokens"] == 10
        assert response.usage["output_tokens"] == 20
    
    async def test_llm_manager_creation(self):
        """Test LLM manager creation and configuration"""
        manager = create_llm_manager()
        
        assert isinstance(manager, LLMManager)
        assert manager.config.provider == LLMProvider.CLAUDE
    
    @patch('core.llm_integration.ClaudeClient.generate')
    async def test_llm_manager_generate(self, mock_generate):
        """Test LLM manager response generation"""
        mock_response = LLMResponse(
            content="Test response",
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            finish_reason="stop",
            response_time=1.5
        )
        mock_generate.return_value = mock_response
        
        manager = create_llm_manager()
        messages = [{"role": "user", "content": "Test"}]
        
        response = await manager.generate(messages, system_prompt="Test system")
        
        assert response.content == "Test response"
        assert response.usage["total_tokens"] == 30


class TestPromptTemplates:
    """Test suite for Prompt Templates functionality"""
    
    def test_prompt_template_creation(self):
        """Test prompt template creation"""
        template = PromptTemplate(
            name="test_template",
            system_template="You are a {role}",
            user_template="Please help with {task}",
            variables={"role": "assistant", "task": "testing"}
        )
        
        assert template.name == "test_template"
        assert "role" in template.variables
        assert "task" in template.variables
    
    def test_prompt_template_formatting(self):
        """Test prompt template formatting"""
        template = PromptTemplate(
            name="test_template",
            system_template="You are a {role}",
            user_template="Please help with {task}",
            variables={"role": "assistant", "task": "testing"}
        )
        
        formatted = template.format(role="legal expert", task="contract analysis")
        
        assert formatted["system"] == "You are a legal expert"
        assert formatted["user"] == "Please help with contract analysis"
    
    def test_legal_prompt_templates_initialization(self):
        """Test legal prompt templates initialization"""
        templates = LegalPromptTemplates()
        
        # Check that templates are loaded
        assert len(templates.templates) > 0
        assert "document_analysis" in templates.templates
        assert "legal_research" in templates.templates
    
    def test_prompt_template_manager(self):
        """Test prompt template manager functionality"""
        manager = PromptTemplateManager()
        
        # Test template creation
        variables = {"question": "What is contract law?"}
        prompt = manager.create_prompt("legal_qa", variables)
        
        assert "system" in prompt
        assert "user" in prompt
        assert "contract law" in prompt["user"]
    
    def test_template_suggestions(self):
        """Test template suggestion functionality"""
        manager = PromptTemplateManager()
        
        suggestions = manager.get_template_suggestions(
            task_type=LegalTaskType.DOCUMENT_ANALYSIS,
            jurisdiction=LegalJurisdiction.VIETNAM
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
    
    def test_get_legal_prompt_function(self):
        """Test get_legal_prompt utility function"""
        prompt = get_legal_prompt(
            task_type=LegalTaskType.LEGAL_RESEARCH,
            question="What are the requirements for employment contracts?",
            jurisdiction=LegalJurisdiction.VIETNAM
        )
        
        assert "system" in prompt
        assert "user" in prompt
        assert "employment contracts" in prompt["user"]


class TestContextManagement:
    """Test suite for Context Management functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.temp_dir, "test_context.db")
    
    def teardown_method(self):
        """Cleanup test environment"""
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
        os.rmdir(self.temp_dir)
    
    async def test_context_storage_initialization(self):
        """Test context storage initialization"""
        storage = SQLiteContextStorage(self.storage_path)
        
        # Check database file creation
        assert os.path.exists(self.storage_path)
        
        # Check table creation
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            
            assert "sessions" in tables
            assert "messages" in tables
            assert "document_contexts" in tables
    
    async def test_session_creation(self):
        """Test conversation session creation"""
        storage = SQLiteContextStorage(self.storage_path)
        
        session_id = "test_session_001"
        user_id = "test_user_123"
        
        session = await storage.create_session(session_id, user_id)
        
        assert session.session_id == session_id
        assert session.user_id == user_id
        assert isinstance(session.created_at, datetime)
    
    async def test_message_storage_and_retrieval(self):
        """Test message storage and retrieval"""
        storage = SQLiteContextStorage(self.storage_path)
        
        session_id = "test_session_001"
        await storage.create_session(session_id, "test_user")
        
        # Add messages
        await storage.add_message(
            session_id, MessageRole.USER, "Hello, I need legal help"
        )
        await storage.add_message(
            session_id, MessageRole.ASSISTANT, "I'm here to help with legal questions"
        )
        
        # Retrieve messages
        messages = await storage.get_conversation_context(session_id, max_messages=10)
        
        assert len(messages) == 2
        assert messages[0]["role"] == MessageRole.USER
        assert messages[1]["role"] == MessageRole.ASSISTANT
        assert "legal help" in messages[0]["content"]
    
    async def test_document_context_management(self):
        """Test document context management"""
        storage = SQLiteContextStorage(self.storage_path)
        
        session_id = "test_session_001"
        await storage.create_session(session_id, "test_user")
        
        # Create document context
        doc_context = create_document_context(
            document_id="doc_001",
            title="Employment Contract",
            content="This is an employment contract...",
            document_type="contract"
        )
        
        # Add document context
        success = await storage.add_document_context(session_id, doc_context)
        assert success
        
        # Retrieve relevant documents
        relevant_docs = await storage.get_relevant_documents(
            session_id, "employment", limit=5
        )
        
        assert len(relevant_docs) > 0
        assert relevant_docs[0].title == "Employment Contract"
    
    async def test_context_manager_integration(self):
        """Test context manager integration"""
        manager = create_context_manager(self.storage_path)
        
        session_id = "test_session_001"
        user_id = "test_user_123"
        
        # Create session
        session = await manager.create_session(session_id, user_id)
        assert session.session_id == session_id
        
        # Add messages
        await manager.add_message(
            session_id, MessageRole.USER, "I need contract analysis"
        )
        
        # Get conversation context
        context = await manager.get_conversation_context(session_id)
        assert len(context) == 1
        assert "contract analysis" in context[0]["content"]


class TestLegalPromptEngine:
    """Test suite for Legal Prompt Engine functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_task_analyzer(self):
        """Test task analyzer functionality"""
        analyzer = TaskAnalyzer()
        
        # Test document analysis detection
        result = analyzer.analyze_query("Please analyze this employment contract")
        assert result["task_type"] == LegalTaskType.DOCUMENT_ANALYSIS
        
        # Test legal research detection
        result = analyzer.analyze_query("I need to research Vietnamese labor law")
        assert result["task_type"] == LegalTaskType.LEGAL_RESEARCH
        assert result["jurisdiction"] == LegalJurisdiction.VIETNAM
        
        # Test compliance check detection
        result = analyzer.analyze_query("Is this contract compliant with regulations?")
        assert result["task_type"] == LegalTaskType.COMPLIANCE_CHECK
    
    def test_legal_query_creation(self):
        """Test legal query structure"""
        query = LegalQuery(
            query_text="Analyze this contract for compliance issues",
            task_type=LegalTaskType.COMPLIANCE_CHECK,
            jurisdiction=LegalJurisdiction.VIETNAM,
            urgency_level="high"
        )
        
        assert query.query_text == "Analyze this contract for compliance issues"
        assert query.task_type == LegalTaskType.COMPLIANCE_CHECK
        assert query.jurisdiction == LegalJurisdiction.VIETNAM
        assert query.urgency_level == "high"
    
    @patch('core.llm_integration.LLMManager.generate')
    async def test_legal_prompt_engine_query_processing(self, mock_generate):
        """Test legal prompt engine query processing"""
        # Mock LLM response
        mock_response = LLMResponse(
            content="Based on my analysis of the contract, here are the key compliance issues...",
            usage={"input_tokens": 100, "output_tokens": 200, "total_tokens": 300},
            finish_reason="stop",
            response_time=2.0
        )
        mock_generate.return_value = mock_response
        
        # Create engine with temporary storage
        engine = create_legal_prompt_engine(storage_path=self.temp_dir)
        
        # Process query
        response = await engine.process_query(
            query_text="Analyze this employment contract for compliance with Vietnamese law",
            session_id="test_session_001",
            user_id="test_user_123"
        )
        
        assert isinstance(response, LegalResponse)
        assert "compliance" in response.content.lower()
        assert response.confidence_score > 0
        assert len(response.follow_up_questions) > 0
    
    async def test_document_addition_to_session(self):
        """Test adding documents to session"""
        engine = create_legal_prompt_engine(storage_path=self.temp_dir)
        
        success = await engine.add_document_to_session(
            session_id="test_session_001",
            document_id="contract_001",
            title="Employment Contract",
            content="This employment agreement is between...",
            document_type="contract",
            summary="Standard employment contract"
        )
        
        assert success


class TestMemorySystem:
    """Test suite for Memory System functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.temp_dir, "test_memory")
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_memory_entry_creation(self):
        """Test memory entry creation"""
        memory = MemoryEntry(
            id="test_memory_001",
            content="User asked about employment contracts",
            memory_type=MemoryType.CONVERSATION,
            importance=MemoryImportance.MEDIUM,
            timestamp=datetime.now(),
            session_id="session_001",
            tags={"employment", "contracts"}
        )
        
        assert memory.id == "test_memory_001"
        assert memory.memory_type == MemoryType.CONVERSATION
        assert memory.importance == MemoryImportance.MEDIUM
        assert "employment" in memory.tags
    
    def test_memory_embedding_system(self):
        """Test memory embedding system"""
        embedding_system = MemoryEmbedding()
        
        # Build vocabulary
        texts = [
            "Employment contract analysis",
            "Legal research on labor law",
            "Contract compliance review"
        ]
        embedding_system._build_vocabulary(texts)
        
        # Test encoding
        embedding = embedding_system.encode("Employment contract")
        assert isinstance(embedding, list)
        assert len(embedding) == embedding_system.dimension
        
        # Test similarity
        embedding1 = embedding_system.encode("Employment contract")
        embedding2 = embedding_system.encode("Labor agreement")
        similarity = embedding_system.similarity(embedding1, embedding2)
        
        assert 0 <= similarity <= 1
    
    async def test_memory_storage(self):
        """Test memory storage functionality"""
        storage = MemoryStorage(self.storage_path)
        
        memory = MemoryEntry(
            id="test_memory_001",
            content="User asked about employment contracts",
            memory_type=MemoryType.CONVERSATION,
            importance=MemoryImportance.MEDIUM,
            timestamp=datetime.now(),
            session_id="session_001"
        )
        
        # Store memory
        success = await storage.store_memory(memory)
        assert success
        
        # Retrieve memory
        retrieved = await storage.retrieve_memory("test_memory_001")
        assert retrieved is not None
        assert retrieved.content == memory.content
        assert retrieved.memory_type == memory.memory_type
    
    async def test_memory_search(self):
        """Test memory search functionality"""
        storage = MemoryStorage(self.storage_path)
        
        # Store test memories
        memories = [
            MemoryEntry(
                id="memory_001",
                content="Employment contract analysis discussion",
                memory_type=MemoryType.CONVERSATION,
                importance=MemoryImportance.HIGH,
                timestamp=datetime.now(),
                session_id="session_001",
                tags={"employment", "contract"}
            ),
            MemoryEntry(
                id="memory_002",
                content="Legal research on termination procedures",
                memory_type=MemoryType.KNOWLEDGE,
                importance=MemoryImportance.MEDIUM,
                timestamp=datetime.now(),
                session_id="session_001",
                tags={"termination", "procedures"}
            )
        ]
        
        for memory in memories:
            await storage.store_memory(memory)
        
        # Search memories
        query = MemoryQuery(
            query_text="employment",
            session_id="session_001",
            limit=5
        )
        
        results = await storage.search_memories(query)
        assert len(results) > 0
        assert any("employment" in result.content.lower() for result in results)
    
    async def test_memory_system_integration(self):
        """Test memory system integration"""
        memory_system = create_memory_system(self.storage_path)
        
        # Store memory
        memory_id = await memory_system.store_memory(
            content="User discussed employment contract termination clauses",
            memory_type=MemoryType.CONVERSATION,
            session_id="session_001",
            importance=MemoryImportance.HIGH,
            tags={"employment", "termination", "clauses"}
        )
        
        assert memory_id is not None
        
        # Retrieve memories
        results = await memory_system.retrieve_memories(
            query_text="employment termination",
            session_id="session_001",
            limit=5
        )
        
        assert len(results) > 0
        assert results[0].memory.content == "User discussed employment contract termination clauses"
        assert results[0].relevance_score > 0


class TestIntegration:
    """Integration tests for complete LLM system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('core.llm_integration.LLMManager.generate')
    async def test_end_to_end_legal_consultation(self, mock_generate):
        """Test complete end-to-end legal consultation workflow"""
        # Mock LLM response
        mock_response = LLMResponse(
            content="Based on Vietnamese labor law, employment contracts must include specific termination procedures. Here are the key requirements...",
            usage={"input_tokens": 150, "output_tokens": 300, "total_tokens": 450},
            finish_reason="stop",
            response_time=2.5
        )
        mock_generate.return_value = mock_response
        
        # Create integrated system
        engine = create_legal_prompt_engine(storage_path=self.temp_dir)
        
        # Simulate consultation workflow
        session_id = "consultation_001"
        user_id = "client_123"
        
        # Step 1: Initial query
        response1 = await engine.process_query(
            query_text="I need help understanding employment contract termination procedures in Vietnam",
            session_id=session_id,
            user_id=user_id
        )
        
        assert "termination procedures" in response1.content.lower()
        assert len(response1.follow_up_questions) > 0
        
        # Step 2: Add document for analysis
        await engine.add_document_to_session(
            session_id=session_id,
            document_id="contract_001",
            title="Employment Contract - Software Developer",
            content="This employment agreement contains termination clauses...",
            document_type="contract"
        )
        
        # Step 3: Follow-up query with document context
        response2 = await engine.process_query(
            query_text="Can you analyze the termination clauses in the uploaded contract?",
            session_id=session_id
        )
        
        assert isinstance(response2, LegalResponse)
        assert response2.confidence_score > 0
        
        # Step 4: Check memory retention
        memory_system = engine.context_manager.storage
        if hasattr(memory_system, 'search_memories'):
            # This would test memory if integrated
            pass


class TestPerformance:
    """Performance tests for LLM integration"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    async def test_prompt_generation_performance(self):
        """Test prompt generation performance"""
        manager = PromptTemplateManager()
        
        start_time = time.time()
        
        # Generate multiple prompts
        for i in range(100):
            variables = {
                "question": f"Legal question {i}",
                "jurisdiction": "Vietnam"
            }
            prompt = manager.create_prompt("legal_qa", variables)
            assert "system" in prompt
            assert "user" in prompt
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete 100 prompt generations in under 1 second
        assert duration < 1.0
    
    async def test_memory_search_performance(self):
        """Test memory search performance"""
        memory_system = create_memory_system(self.storage_path)
        
        # Store multiple memories
        for i in range(100):
            await memory_system.store_memory(
                content=f"Legal discussion about topic {i}",
                memory_type=MemoryType.CONVERSATION,
                session_id=f"session_{i % 10}",
                importance=MemoryImportance.MEDIUM,
                tags={f"topic_{i}", "legal", "discussion"}
            )
        
        # Test search performance
        start_time = time.time()
        
        results = await memory_system.retrieve_memories(
            query_text="legal discussion",
            limit=10
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete search in under 1 second
        assert duration < 1.0
        assert len(results) > 0
    
    async def test_context_retrieval_performance(self):
        """Test context retrieval performance"""
        manager = create_context_manager(self.temp_dir)
        
        session_id = "performance_test_session"
        await manager.create_session(session_id, "test_user")
        
        # Add many messages
        for i in range(1000):
            await manager.add_message(
                session_id, 
                MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                f"Message content {i}"
            )
        
        # Test retrieval performance
        start_time = time.time()
        
        context = await manager.get_conversation_context(session_id, max_messages=50)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should retrieve context in under 0.5 seconds
        assert duration < 0.5
        assert len(context) == 50


# Test configuration and fixtures
@pytest.fixture
def temp_storage():
    """Provide temporary storage for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture
async def mock_llm_manager():
    """Provide mock LLM manager for testing"""
    manager = Mock(spec=LLMManager)
    manager.generate = AsyncMock(return_value=LLMResponse(
        content="Mock response",
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        finish_reason="stop",
        response_time=1.0
    ))
    return manager


# Test runner configuration
if __name__ == "__main__":
    # Run specific test categories
    import pytest
    
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Run specific test class
    # pytest.main([f"{__file__}::TestLLMIntegration", "-v"])
    
    # Run performance tests only
    # pytest.main([f"{__file__}::TestPerformance", "-v"])
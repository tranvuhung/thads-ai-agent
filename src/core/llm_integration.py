"""
Core LLM Integration Module

This module provides the foundation for integrating Large Language Models (LLMs),
specifically Claude, into the legal AI agent system. It includes:
- Claude API client with robust error handling
- Configuration management for different LLM providers
- Base classes for LLM operations
- Response processing and validation
- Rate limiting and retry mechanisms
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, AsyncGenerator
import os
from datetime import datetime, timedelta

try:
    import anthropic
    from anthropic import Anthropic, AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None
    Anthropic = None
    AsyncAnthropic = None

# Configure logging
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    CLAUDE = "claude"
    OPENAI = "openai"
    LOCAL = "local"


class ModelType(Enum):
    """Available model types for each provider"""
    # Claude models
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    
    # OpenAI models (for future expansion)
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo"


class ResponseFormat(Enum):
    """Response format options"""
    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"


@dataclass
class LLMConfig:
    """Configuration for LLM integration"""
    provider: LLMProvider
    model: ModelType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 100000
    enable_streaming: bool = False
    response_format: ResponseFormat = ResponseFormat.TEXT
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.api_key:
            if self.provider == LLMProvider.CLAUDE:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == LLMProvider.OPENAI:
                self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key and self.provider != LLMProvider.LOCAL:
            raise ValueError(f"API key required for {self.provider.value}")


@dataclass
class LLMRequest:
    """Request structure for LLM calls"""
    messages: List[Dict[str, str]]
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stream: bool = False


@dataclass
class LLMResponse:
    """Response structure from LLM calls"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    response_time: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "response_time": self.response_time,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_times: List[float] = []
        self.token_usage: List[tuple] = []  # (timestamp, tokens)
        
    async def wait_if_needed(self, estimated_tokens: int = 0):
        """Wait if rate limits would be exceeded"""
        current_time = time.time()
        
        # Clean old entries
        cutoff_time = current_time - 60  # 1 minute ago
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff_time]
        
        # Check request rate limit
        if len(self.request_times) >= self.requests_per_minute:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Check token rate limit
        current_tokens = sum(tokens for _, tokens in self.token_usage)
        if current_tokens + estimated_tokens > self.tokens_per_minute:
            wait_time = 60 - (current_time - self.token_usage[0][0])
            if wait_time > 0:
                logger.info(f"Token rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.request_times.append(current_time)
        if estimated_tokens > 0:
            self.token_usage.append((current_time, estimated_tokens))


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.rate_limiter = RateLimiter(
            config.rate_limit_requests_per_minute,
            config.rate_limit_tokens_per_minute
        )
    
    @abstractmethod
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response from LLM"""
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        pass


class ClaudeClient(BaseLLMClient):
    """Claude API client implementation"""
    
    def __init__(self, config: LLMConfig):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required for Claude integration")
        
        super().__init__(config)
        self.client = Anthropic(api_key=config.api_key)
        self.async_client = AsyncAnthropic(api_key=config.api_key)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
        return len(text) // 4
    
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response from Claude"""
        start_time = time.time()
        
        # Estimate tokens for rate limiting
        total_text = ""
        if request.system_prompt:
            total_text += request.system_prompt
        for message in request.messages:
            total_text += message.get("content", "")
        
        estimated_tokens = self.estimate_tokens(total_text)
        await self.rate_limiter.wait_if_needed(estimated_tokens)
        
        # Prepare request parameters
        kwargs = {
            "model": self.config.model.value,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "temperature": request.temperature or self.config.temperature,
            "top_p": request.top_p or self.config.top_p,
            "messages": request.messages
        }
        
        if request.system_prompt:
            kwargs["system"] = request.system_prompt
        
        if request.stop_sequences:
            kwargs["stop_sequences"] = request.stop_sequences
        
        # Make API call with retries
        for attempt in range(self.config.max_retries):
            try:
                response = await self.async_client.messages.create(**kwargs)
                
                response_time = time.time() - start_time
                
                return LLMResponse(
                    content=response.content[0].text,
                    model=response.model,
                    usage={
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                    },
                    finish_reason=response.stop_reason,
                    response_time=response_time,
                    timestamp=datetime.now(),
                    metadata=request.metadata
                )
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
    
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response from Claude"""
        # Estimate tokens for rate limiting
        total_text = ""
        if request.system_prompt:
            total_text += request.system_prompt
        for message in request.messages:
            total_text += message.get("content", "")
        
        estimated_tokens = self.estimate_tokens(total_text)
        await self.rate_limiter.wait_if_needed(estimated_tokens)
        
        # Prepare request parameters
        kwargs = {
            "model": self.config.model.value,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "temperature": request.temperature or self.config.temperature,
            "top_p": request.top_p or self.config.top_p,
            "messages": request.messages,
            "stream": True
        }
        
        if request.system_prompt:
            kwargs["system"] = request.system_prompt
        
        if request.stop_sequences:
            kwargs["stop_sequences"] = request.stop_sequences
        
        # Make streaming API call
        async with self.async_client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text


class LLMManager:
    """Manager class for LLM operations"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._create_client()
        self.request_history: List[Dict[str, Any]] = []
    
    def _create_client(self) -> BaseLLMClient:
        """Create appropriate LLM client based on configuration"""
        if self.config.provider == LLMProvider.CLAUDE:
            return ClaudeClient(self.config)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response with conversation history tracking"""
        request = LLMRequest(
            messages=messages,
            system_prompt=system_prompt,
            **kwargs
        )
        
        response = await self.client.generate_response(request)
        
        # Track request/response for debugging
        self.request_history.append({
            "request": {
                "messages": messages,
                "system_prompt": system_prompt,
                "timestamp": datetime.now().isoformat()
            },
            "response": response.to_dict()
        })
        
        # Keep only last 100 requests
        if len(self.request_history) > 100:
            self.request_history = self.request_history[-100:]
        
        return response
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        request = LLMRequest(
            messages=messages,
            system_prompt=system_prompt,
            stream=True,
            **kwargs
        )
        
        async for chunk in self.client.generate_stream(request):
            yield chunk
    
    def get_request_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent request history"""
        return self.request_history[-limit:]
    
    def clear_history(self):
        """Clear request history"""
        self.request_history.clear()


# Factory functions for easy configuration
def create_claude_config(
    model: ModelType = ModelType.CLAUDE_3_5_SONNET,
    api_key: Optional[str] = None,
    **kwargs
) -> LLMConfig:
    """Create Claude configuration with sensible defaults"""
    return LLMConfig(
        provider=LLMProvider.CLAUDE,
        model=model,
        api_key=api_key,
        **kwargs
    )


def create_llm_manager(
    provider: LLMProvider = LLMProvider.CLAUDE,
    model: ModelType = ModelType.CLAUDE_3_5_SONNET,
    **kwargs
) -> LLMManager:
    """Create LLM manager with default configuration"""
    if provider == LLMProvider.CLAUDE:
        config = create_claude_config(model=model, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    return LLMManager(config)


# Example usage and testing functions
async def test_claude_integration():
    """Test Claude integration"""
    try:
        manager = create_llm_manager()
        
        messages = [
            {"role": "user", "content": "Hello, can you help me with legal document analysis?"}
        ]
        
        response = await manager.generate(
            messages=messages,
            system_prompt="You are a helpful legal AI assistant."
        )
        
        print(f"Response: {response.content}")
        print(f"Tokens used: {response.usage}")
        print(f"Response time: {response.response_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"Claude integration test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run basic test
    asyncio.run(test_claude_integration())
"""
Legal Prompt Engine

This module provides an intelligent prompt engine specifically designed for legal AI tasks.
It combines LLM integration, prompt templates, and context management to deliver
sophisticated legal assistance with proper context awareness and domain expertise.

Key Features:
- Intelligent template selection based on task analysis
- Context-aware prompt generation
- Legal domain knowledge integration
- Multi-turn conversation support
- Document-aware prompting
- Compliance and risk assessment integration
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
import re
from pathlib import Path

from .llm_integration import LLMManager, LLMResponse, ModelType, create_llm_manager
from .prompt_templates import (
    LegalPromptTemplates, PromptTemplateManager, LegalTaskType, 
    LegalJurisdiction, DocumentType, get_legal_prompt
)
from .context_management import (
    ContextManager, ConversationSession, DocumentContext, 
    MessageRole, ContextEntry, ContextType, ContextPriority,
    create_context_manager, create_document_context
)

# Configure logging
logger = logging.getLogger(__name__)


class PromptStrategy(Enum):
    """Strategies for prompt generation"""
    DIRECT = "direct"  # Direct template application
    CONTEXT_AWARE = "context_aware"  # Include conversation context
    DOCUMENT_FOCUSED = "document_focused"  # Focus on document analysis
    MULTI_TURN = "multi_turn"  # Multi-turn conversation optimization
    RESEARCH_ORIENTED = "research_oriented"  # Legal research focus
    COMPLIANCE_FOCUSED = "compliance_focused"  # Compliance and risk focus


class ResponseMode(Enum):
    """Response generation modes"""
    STANDARD = "standard"  # Standard response
    DETAILED = "detailed"  # Detailed analysis
    SUMMARY = "summary"  # Concise summary
    STEP_BY_STEP = "step_by_step"  # Step-by-step guidance
    COMPARATIVE = "comparative"  # Comparative analysis
    RISK_ASSESSMENT = "risk_assessment"  # Risk-focused response


@dataclass
class LegalQuery:
    """Structured legal query with context"""
    query_text: str
    task_type: Optional[LegalTaskType] = None
    jurisdiction: Optional[LegalJurisdiction] = None
    document_type: Optional[DocumentType] = None
    urgency_level: str = "medium"
    response_mode: ResponseMode = ResponseMode.STANDARD
    context_requirements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary"""
        return {
            "query_text": self.query_text,
            "task_type": self.task_type.value if self.task_type else None,
            "jurisdiction": self.jurisdiction.value if self.jurisdiction else None,
            "document_type": self.document_type.value if self.document_type else None,
            "urgency_level": self.urgency_level,
            "response_mode": self.response_mode.value,
            "context_requirements": self.context_requirements,
            "metadata": self.metadata
        }


@dataclass
class PromptGenerationResult:
    """Result of prompt generation"""
    system_prompt: str
    user_prompt: str
    template_used: str
    strategy_applied: PromptStrategy
    context_included: Dict[str, Any]
    estimated_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LegalResponse:
    """Enhanced legal response with metadata"""
    content: str
    llm_response: LLMResponse
    query: LegalQuery
    prompt_result: PromptGenerationResult
    confidence_score: float = 0.0
    citations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    risk_indicators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            "content": self.content,
            "llm_response": self.llm_response.to_dict(),
            "query": self.query.to_dict(),
            "confidence_score": self.confidence_score,
            "citations": self.citations,
            "recommendations": self.recommendations,
            "follow_up_questions": self.follow_up_questions,
            "risk_indicators": self.risk_indicators,
            "metadata": {
                "template_used": self.prompt_result.template_used,
                "strategy_applied": self.prompt_result.strategy_applied.value,
                "estimated_tokens": self.prompt_result.estimated_tokens
            }
        }


class TaskAnalyzer:
    """Analyzes queries to determine appropriate task types and strategies"""
    
    def __init__(self):
        self.task_keywords = {
            LegalTaskType.DOCUMENT_ANALYSIS: [
                "analyze", "review", "examine", "assess", "evaluate", "document", "contract", "agreement"
            ],
            LegalTaskType.LEGAL_RESEARCH: [
                "research", "find", "law", "regulation", "statute", "precedent", "case law", "legal basis"
            ],
            LegalTaskType.CONTRACT_REVIEW: [
                "contract", "agreement", "terms", "conditions", "clause", "liability", "obligation"
            ],
            LegalTaskType.COMPLIANCE_CHECK: [
                "compliance", "compliant", "regulation", "requirement", "standard", "audit", "violation"
            ],
            LegalTaskType.LEGAL_QA: [
                "what", "how", "when", "where", "why", "can", "should", "must", "question", "explain"
            ],
            LegalTaskType.RISK_ASSESSMENT: [
                "risk", "liability", "exposure", "danger", "threat", "vulnerability", "assessment"
            ]
        }
        
        self.jurisdiction_keywords = {
            LegalJurisdiction.VIETNAM: [
                "vietnam", "vietnamese", "vn", "hanoi", "ho chi minh", "saigon"
            ],
            LegalJurisdiction.US_FEDERAL: [
                "us", "usa", "united states", "federal", "american"
            ],
            LegalJurisdiction.EU: [
                "eu", "european", "europe", "gdpr", "european union"
            ]
        }
        
        self.document_keywords = {
            DocumentType.CONTRACT: [
                "contract", "agreement", "deal", "terms", "conditions"
            ],
            DocumentType.REGULATION: [
                "regulation", "rule", "directive", "order", "decree"
            ],
            DocumentType.STATUTE: [
                "law", "statute", "act", "code", "legislation"
            ]
        }
    
    def analyze_query(self, query_text: str) -> Dict[str, Any]:
        """Analyze query to determine task type, jurisdiction, and other attributes"""
        query_lower = query_text.lower()
        
        # Determine task type
        task_scores = {}
        for task_type, keywords in self.task_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                task_scores[task_type] = score
        
        best_task = max(task_scores.items(), key=lambda x: x[1])[0] if task_scores else None
        
        # Determine jurisdiction
        jurisdiction_scores = {}
        for jurisdiction, keywords in self.jurisdiction_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                jurisdiction_scores[jurisdiction] = score
        
        best_jurisdiction = max(jurisdiction_scores.items(), key=lambda x: x[1])[0] if jurisdiction_scores else LegalJurisdiction.VIETNAM
        
        # Determine document type
        document_scores = {}
        for doc_type, keywords in self.document_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                document_scores[doc_type] = score
        
        best_document_type = max(document_scores.items(), key=lambda x: x[1])[0] if document_scores else None
        
        # Determine urgency
        urgency = "medium"
        if any(word in query_lower for word in ["urgent", "asap", "immediately", "emergency"]):
            urgency = "high"
        elif any(word in query_lower for word in ["when convenient", "no rush", "eventually"]):
            urgency = "low"
        
        # Determine response mode
        response_mode = ResponseMode.STANDARD
        if any(word in query_lower for word in ["detailed", "comprehensive", "thorough"]):
            response_mode = ResponseMode.DETAILED
        elif any(word in query_lower for word in ["summary", "brief", "concise"]):
            response_mode = ResponseMode.SUMMARY
        elif any(word in query_lower for word in ["step", "guide", "how to"]):
            response_mode = ResponseMode.STEP_BY_STEP
        elif any(word in query_lower for word in ["compare", "versus", "difference"]):
            response_mode = ResponseMode.COMPARATIVE
        elif any(word in query_lower for word in ["risk", "danger", "liability"]):
            response_mode = ResponseMode.RISK_ASSESSMENT
        
        return {
            "task_type": best_task,
            "jurisdiction": best_jurisdiction,
            "document_type": best_document_type,
            "urgency_level": urgency,
            "response_mode": response_mode,
            "confidence": max(task_scores.values()) if task_scores else 0
        }


class ContextEnricher:
    """Enriches prompts with relevant context information"""
    
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
    
    async def enrich_prompt(
        self, 
        base_prompt: Dict[str, str],
        session_id: str,
        query: LegalQuery,
        strategy: PromptStrategy
    ) -> Dict[str, Any]:
        """Enrich prompt with context information"""
        enriched_context = {
            "conversation_history": [],
            "relevant_documents": [],
            "session_summary": {},
            "context_entries": []
        }
        
        # Get conversation context
        if strategy in [PromptStrategy.CONTEXT_AWARE, PromptStrategy.MULTI_TURN]:
            conversation_context = await self.context_manager.get_conversation_context(
                session_id, max_messages=10
            )
            enriched_context["conversation_history"] = conversation_context
        
        # Get relevant documents
        if strategy in [PromptStrategy.DOCUMENT_FOCUSED, PromptStrategy.CONTEXT_AWARE]:
            relevant_docs = await self.context_manager.get_relevant_documents(
                session_id, query.query_text, limit=3
            )
            enriched_context["relevant_documents"] = [
                {
                    "title": doc.title,
                    "type": doc.document_type,
                    "summary": doc.summary or doc.content[:200] + "...",
                    "relevance_score": doc.relevance_score
                }
                for doc in relevant_docs
            ]
        
        # Get session summary
        session_summary = await self.context_manager.get_session_summary(session_id)
        enriched_context["session_summary"] = session_summary
        
        # Enhance system prompt with context
        enhanced_system_prompt = base_prompt["system"]
        
        if enriched_context["conversation_history"]:
            enhanced_system_prompt += f"\n\nConversation Context:\nYou are continuing a conversation. Previous messages: {len(enriched_context['conversation_history'])} messages exchanged."
        
        if enriched_context["relevant_documents"]:
            doc_context = "\n\nRelevant Documents Available:\n"
            for doc in enriched_context["relevant_documents"]:
                doc_context += f"- {doc['title']} ({doc['type']}): {doc['summary']}\n"
            enhanced_system_prompt += doc_context
        
        # Enhance user prompt with context
        enhanced_user_prompt = base_prompt["user"]
        
        if strategy == PromptStrategy.MULTI_TURN and enriched_context["conversation_history"]:
            conversation_summary = "\n\nConversation History:\n"
            for msg in enriched_context["conversation_history"][-5:]:  # Last 5 messages
                conversation_summary += f"{msg['role'].upper()}: {msg['content'][:100]}...\n"
            enhanced_user_prompt = conversation_summary + "\n\nCurrent Query:\n" + enhanced_user_prompt
        
        return {
            "system": enhanced_system_prompt,
            "user": enhanced_user_prompt,
            "context": enriched_context
        }


class LegalPromptEngine:
    """Main legal prompt engine that orchestrates all components"""
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        context_manager: Optional[ContextManager] = None,
        prompt_manager: Optional[PromptTemplateManager] = None
    ):
        self.llm_manager = llm_manager or create_llm_manager()
        self.context_manager = context_manager or create_context_manager()
        self.prompt_manager = prompt_manager or PromptTemplateManager()
        self.task_analyzer = TaskAnalyzer()
        self.context_enricher = ContextEnricher(self.context_manager)
        
        # Performance tracking
        self.query_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "total_queries": 0,
            "successful_responses": 0,
            "average_response_time": 0.0,
            "template_usage": {}
        }
    
    async def process_query(
        self,
        query_text: str,
        session_id: str,
        user_id: Optional[str] = None,
        strategy: Optional[PromptStrategy] = None,
        **kwargs
    ) -> LegalResponse:
        """Process a legal query and generate response"""
        start_time = datetime.now()
        
        try:
            # Ensure session exists
            session = await self.context_manager.get_session(session_id)
            if not session:
                session = await self.context_manager.create_session(session_id, user_id)
            
            # Analyze query
            analysis_result = self.task_analyzer.analyze_query(query_text)
            
            # Create structured query
            query = LegalQuery(
                query_text=query_text,
                task_type=analysis_result.get("task_type"),
                jurisdiction=analysis_result.get("jurisdiction"),
                document_type=analysis_result.get("document_type"),
                urgency_level=analysis_result.get("urgency_level", "medium"),
                response_mode=analysis_result.get("response_mode", ResponseMode.STANDARD),
                metadata=kwargs
            )
            
            # Determine strategy
            if not strategy:
                strategy = self._determine_strategy(query, session)
            
            # Generate prompt
            prompt_result = await self._generate_prompt(query, session_id, strategy)
            
            # Add user message to context
            await self.context_manager.add_message(
                session_id, MessageRole.USER, query_text
            )
            
            # Generate LLM response
            messages = [
                {"role": "user", "content": prompt_result.user_prompt}
            ]
            
            llm_response = await self.llm_manager.generate(
                messages=messages,
                system_prompt=prompt_result.system_prompt
            )
            
            # Add assistant response to context
            await self.context_manager.add_message(
                session_id, 
                MessageRole.ASSISTANT, 
                llm_response.content,
                tokens_used=llm_response.usage.get("total_tokens"),
                response_time=llm_response.response_time
            )
            
            # Process and enhance response
            legal_response = await self._process_response(
                llm_response, query, prompt_result
            )
            
            # Update performance metrics
            self._update_metrics(start_time, True, prompt_result.template_used)
            
            # Log query for analysis
            self.query_history.append({
                "timestamp": start_time.isoformat(),
                "query": query.to_dict(),
                "strategy": strategy.value,
                "template_used": prompt_result.template_used,
                "response_time": (datetime.now() - start_time).total_seconds(),
                "success": True
            })
            
            return legal_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            self._update_metrics(start_time, False, "error")
            raise
    
    def _determine_strategy(self, query: LegalQuery, session: ConversationSession) -> PromptStrategy:
        """Determine the best strategy for the query"""
        # Check if we have conversation history
        has_history = len(session.messages) > 0
        
        # Check if we have relevant documents
        has_documents = len(session.document_contexts) > 0
        
        # Determine strategy based on query and context
        if query.task_type == LegalTaskType.DOCUMENT_ANALYSIS and has_documents:
            return PromptStrategy.DOCUMENT_FOCUSED
        elif query.task_type == LegalTaskType.LEGAL_RESEARCH:
            return PromptStrategy.RESEARCH_ORIENTED
        elif query.task_type == LegalTaskType.COMPLIANCE_CHECK:
            return PromptStrategy.COMPLIANCE_FOCUSED
        elif has_history:
            return PromptStrategy.MULTI_TURN
        elif has_documents:
            return PromptStrategy.CONTEXT_AWARE
        else:
            return PromptStrategy.DIRECT
    
    async def _generate_prompt(
        self, 
        query: LegalQuery, 
        session_id: str, 
        strategy: PromptStrategy
    ) -> PromptGenerationResult:
        """Generate prompt based on query and strategy"""
        # Select appropriate template
        template_name = self._select_template(query)
        
        # Prepare template variables
        template_vars = self._prepare_template_variables(query)
        
        # Generate base prompt
        base_prompt = self.prompt_manager.create_prompt(template_name, template_vars)
        
        # Enrich with context
        enriched_prompt = await self.context_enricher.enrich_prompt(
            base_prompt, session_id, query, strategy
        )
        
        # Estimate tokens (rough approximation)
        estimated_tokens = (
            len(enriched_prompt["system"]) + len(enriched_prompt["user"])
        ) // 4
        
        return PromptGenerationResult(
            system_prompt=enriched_prompt["system"],
            user_prompt=enriched_prompt["user"],
            template_used=template_name,
            strategy_applied=strategy,
            context_included=enriched_prompt["context"],
            estimated_tokens=estimated_tokens
        )
    
    def _select_template(self, query: LegalQuery) -> str:
        """Select the most appropriate template for the query"""
        if query.task_type:
            suggestions = self.prompt_manager.get_template_suggestions(
                task_type=query.task_type,
                jurisdiction=query.jurisdiction,
                document_type=query.document_type
            )
            if suggestions:
                return suggestions[0]
        
        # Default fallback
        return "legal_qa"
    
    def _prepare_template_variables(self, query: LegalQuery) -> Dict[str, str]:
        """Prepare variables for template formatting"""
        variables = {
            "question": query.query_text,
            "legal_area": "General Legal",
            "jurisdiction": query.jurisdiction.value if query.jurisdiction else "Vietnam",
            "context": "Legal consultation request",
            "questioner_background": "Legal consultation client",
            "urgency_level": query.urgency_level
        }
        
        # Add task-specific variables
        if query.task_type == LegalTaskType.DOCUMENT_ANALYSIS:
            variables.update({
                "document_type": query.document_type.value if query.document_type else "Legal Document",
                "document_title": "Document for Analysis",
                "document_content": "[Document content will be provided]",
                "analysis_requirements": f"Provide {query.response_mode.value} analysis"
            })
        elif query.task_type == LegalTaskType.LEGAL_RESEARCH:
            variables.update({
                "research_question": query.query_text,
                "specific_focus": "Comprehensive legal analysis",
                "additional_context": "Legal research request",
                "research_scope": "Current laws and regulations"
            })
        
        return variables
    
    async def _process_response(
        self, 
        llm_response: LLMResponse, 
        query: LegalQuery, 
        prompt_result: PromptGenerationResult
    ) -> LegalResponse:
        """Process and enhance the LLM response"""
        content = llm_response.content
        
        # Extract citations (simple regex-based extraction)
        citations = re.findall(r'\[([^\]]+)\]', content)
        
        # Extract recommendations (look for bullet points or numbered lists)
        recommendations = []
        recommendation_patterns = [
            r'(?:Recommendation|Suggest|Advise)s?:?\s*(.+?)(?:\n|$)',
            r'(?:•|\*|\d+\.)\s*(.+?)(?:\n|$)'
        ]
        for pattern in recommendation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            recommendations.extend(matches)
        
        # Extract risk indicators
        risk_indicators = []
        risk_patterns = [
            r'(?:Risk|Danger|Warning|Caution):?\s*(.+?)(?:\n|$)',
            r'(?:potential|possible|may|could)\s+(?:risk|issue|problem)\s*(.+?)(?:\n|$)'
        ]
        for pattern in risk_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            risk_indicators.extend(matches)
        
        # Generate follow-up questions
        follow_up_questions = self._generate_follow_up_questions(query, content)
        
        # Calculate confidence score (simple heuristic)
        confidence_score = self._calculate_confidence_score(llm_response, query)
        
        return LegalResponse(
            content=content,
            llm_response=llm_response,
            query=query,
            prompt_result=prompt_result,
            confidence_score=confidence_score,
            citations=citations[:5],  # Limit to top 5
            recommendations=recommendations[:3],  # Limit to top 3
            follow_up_questions=follow_up_questions,
            risk_indicators=risk_indicators[:3]  # Limit to top 3
        )
    
    def _generate_follow_up_questions(self, query: LegalQuery, response_content: str) -> List[str]:
        """Generate relevant follow-up questions"""
        follow_ups = []
        
        if query.task_type == LegalTaskType.DOCUMENT_ANALYSIS:
            follow_ups = [
                "Would you like me to analyze any specific clauses in more detail?",
                "Do you need recommendations for improving this document?",
                "Are there any compliance concerns you'd like me to address?"
            ]
        elif query.task_type == LegalTaskType.LEGAL_RESEARCH:
            follow_ups = [
                "Would you like me to research related case law?",
                "Do you need information about recent regulatory changes?",
                "Should I provide comparative analysis with other jurisdictions?"
            ]
        elif query.task_type == LegalTaskType.COMPLIANCE_CHECK:
            follow_ups = [
                "Would you like a detailed compliance implementation plan?",
                "Do you need help with compliance monitoring procedures?",
                "Should I assess compliance risks for your specific situation?"
            ]
        else:
            follow_ups = [
                "Would you like more detailed information on any specific aspect?",
                "Do you have additional questions about this topic?",
                "Would you like me to analyze any related documents?"
            ]
        
        return follow_ups[:3]  # Return top 3
    
    def _calculate_confidence_score(self, llm_response: LLMResponse, query: LegalQuery) -> float:
        """Calculate confidence score for the response"""
        score = 0.5  # Base score
        
        # Adjust based on response length (longer responses often more comprehensive)
        if len(llm_response.content) > 500:
            score += 0.1
        if len(llm_response.content) > 1000:
            score += 0.1
        
        # Adjust based on finish reason
        if llm_response.finish_reason == "stop":
            score += 0.2
        
        # Adjust based on response time (faster might indicate more confident)
        if llm_response.response_time < 5.0:
            score += 0.1
        
        return min(1.0, score)
    
    def _update_metrics(self, start_time: datetime, success: bool, template_used: str):
        """Update performance metrics"""
        self.performance_metrics["total_queries"] += 1
        
        if success:
            self.performance_metrics["successful_responses"] += 1
            
            # Update average response time
            response_time = (datetime.now() - start_time).total_seconds()
            current_avg = self.performance_metrics["average_response_time"]
            total_queries = self.performance_metrics["total_queries"]
            
            self.performance_metrics["average_response_time"] = (
                (current_avg * (total_queries - 1) + response_time) / total_queries
            )
            
            # Update template usage
            if template_used not in self.performance_metrics["template_usage"]:
                self.performance_metrics["template_usage"][template_used] = 0
            self.performance_metrics["template_usage"][template_used] += 1
    
    async def add_document_to_session(
        self, 
        session_id: str, 
        document_id: str,
        title: str,
        content: str,
        document_type: str,
        summary: Optional[str] = None
    ) -> bool:
        """Add a document to the session context"""
        doc_context = create_document_context(
            document_id=document_id,
            title=title,
            content=content,
            document_type=document_type,
            summary=summary
        )
        
        return await self.context_manager.add_document_context(session_id, doc_context)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_metrics.copy()
    
    def get_query_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent query history"""
        return self.query_history[-limit:]


# Factory function
def create_legal_prompt_engine(
    model_type: ModelType = ModelType.CLAUDE_3_5_SONNET,
    storage_path: Optional[str] = None
) -> LegalPromptEngine:
    """Create a legal prompt engine with default configuration"""
    llm_manager = create_llm_manager(model=model_type)
    context_manager = create_context_manager(storage_path)
    prompt_manager = PromptTemplateManager()
    
    return LegalPromptEngine(llm_manager, context_manager, prompt_manager)


# Example usage
async def example_usage():
    """Example of how to use the legal prompt engine"""
    # Create the engine
    engine = create_legal_prompt_engine()
    
    # Process a legal query
    response = await engine.process_query(
        query_text="I need help analyzing an employment contract for compliance with Vietnamese labor law",
        session_id="session_001",
        user_id="user_123"
    )
    
    print("Response:", response.content)
    print("Confidence:", response.confidence_score)
    print("Recommendations:", response.recommendations)
    print("Follow-up questions:", response.follow_up_questions)
    
    # Add a document to the session
    await engine.add_document_to_session(
        session_id="session_001",
        document_id="contract_001",
        title="Employment Contract - Software Developer",
        content="This employment agreement is between...",
        document_type="contract",
        summary="Standard employment contract with IP clauses"
    )
    
    # Process another query with document context
    response2 = await engine.process_query(
        query_text="What are the potential risks in the termination clauses?",
        session_id="session_001"
    )
    
    print("\nSecond response:", response2.content)
    print("Risk indicators:", response2.risk_indicators)


if __name__ == "__main__":
    asyncio.run(example_usage())
"""
Legal Domain Prompt Templates

This module provides specialized prompt templates for legal AI tasks including:
- Document analysis and summarization
- Legal research and case law analysis
- Contract review and drafting
- Compliance checking and regulatory analysis
- Legal question answering
- Citation and reference formatting
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json
from datetime import datetime


class LegalTaskType(Enum):
    """Types of legal tasks supported by prompt templates"""
    DOCUMENT_ANALYSIS = "document_analysis"
    LEGAL_RESEARCH = "legal_research"
    CONTRACT_REVIEW = "contract_review"
    COMPLIANCE_CHECK = "compliance_check"
    LEGAL_QA = "legal_qa"
    CASE_ANALYSIS = "case_analysis"
    REGULATION_ANALYSIS = "regulation_analysis"
    LEGAL_DRAFTING = "legal_drafting"
    CITATION_FORMATTING = "citation_formatting"
    RISK_ASSESSMENT = "risk_assessment"


class LegalJurisdiction(Enum):
    """Legal jurisdictions for context-specific prompts"""
    VIETNAM = "vietnam"
    US_FEDERAL = "us_federal"
    EU = "european_union"
    INTERNATIONAL = "international"
    COMMON_LAW = "common_law"
    CIVIL_LAW = "civil_law"


class DocumentType(Enum):
    """Types of legal documents"""
    CONTRACT = "contract"
    REGULATION = "regulation"
    CASE_LAW = "case_law"
    STATUTE = "statute"
    LEGAL_OPINION = "legal_opinion"
    BRIEF = "brief"
    MOTION = "motion"
    AGREEMENT = "agreement"
    POLICY = "policy"
    COMPLIANCE_REPORT = "compliance_report"


@dataclass
class PromptTemplate:
    """Base class for prompt templates"""
    name: str
    task_type: LegalTaskType
    system_prompt: str
    user_prompt_template: str
    jurisdiction: Optional[LegalJurisdiction] = None
    document_type: Optional[DocumentType] = None
    variables: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def format_prompt(self, **kwargs) -> Dict[str, str]:
        """Format the prompt template with provided variables"""
        # Validate required variables
        missing_vars = [var for var in self.variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Format user prompt
        user_prompt = self.user_prompt_template.format(**kwargs)
        
        return {
            "system": self.system_prompt,
            "user": user_prompt
        }


class LegalPromptTemplates:
    """Collection of legal domain prompt templates"""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize all legal prompt templates"""
        
        # Document Analysis Template
        self.templates["document_analysis"] = PromptTemplate(
            name="Legal Document Analysis",
            task_type=LegalTaskType.DOCUMENT_ANALYSIS,
            system_prompt="""You are an expert legal analyst specializing in Vietnamese law and international legal frameworks. Your role is to provide comprehensive, accurate, and actionable analysis of legal documents.

Key responsibilities:
- Analyze legal documents with precision and attention to detail
- Identify key legal concepts, obligations, rights, and risks
- Provide clear, structured summaries accessible to both legal professionals and non-lawyers
- Highlight potential issues, ambiguities, or areas requiring attention
- Reference relevant Vietnamese laws, regulations, and legal precedents when applicable
- Maintain objectivity and provide balanced analysis

Analysis Framework:
1. Document Overview and Purpose
2. Key Legal Provisions and Clauses
3. Rights and Obligations of Parties
4. Potential Legal Risks and Issues
5. Compliance Requirements
6. Recommendations and Next Steps""",
            user_prompt_template="""Please analyze the following legal document:

**Document Type:** {document_type}
**Document Title:** {document_title}
**Jurisdiction:** {jurisdiction}

**Document Content:**
{document_content}

**Specific Analysis Requirements:**
{analysis_requirements}

Please provide a comprehensive analysis following the framework outlined in your instructions. Focus on practical implications and actionable insights.""",
            jurisdiction=LegalJurisdiction.VIETNAM,
            variables=["document_type", "document_title", "jurisdiction", "document_content", "analysis_requirements"],
            examples=[
                {
                    "input": "Contract analysis for employment agreement",
                    "output": "Comprehensive analysis with risk assessment and compliance check"
                }
            ]
        )
        
        # Legal Research Template
        self.templates["legal_research"] = PromptTemplate(
            name="Legal Research Assistant",
            task_type=LegalTaskType.LEGAL_RESEARCH,
            system_prompt="""You are a specialized legal research assistant with expertise in Vietnamese law, international law, and comparative legal analysis. Your role is to conduct thorough legal research and provide comprehensive, well-cited responses.

Research Methodology:
- Identify relevant legal sources (statutes, regulations, case law, legal doctrine)
- Analyze legal precedents and their applicability
- Consider both current law and recent developments
- Provide comparative analysis when relevant
- Cite sources accurately and comprehensively

Research Framework:
1. Legal Issue Identification and Scope
2. Applicable Legal Framework
3. Relevant Statutes and Regulations
4. Case Law and Precedents
5. Legal Doctrine and Commentary
6. Practical Applications and Implications
7. Recent Developments and Trends
8. Recommendations and Conclusions""",
            user_prompt_template="""Please conduct legal research on the following topic:

**Research Question:** {research_question}
**Legal Area:** {legal_area}
**Jurisdiction:** {jurisdiction}
**Specific Focus:** {specific_focus}

**Additional Context:**
{additional_context}

**Research Scope:**
{research_scope}

Please provide comprehensive research following the framework outlined, with proper citations and practical insights.""",
            jurisdiction=LegalJurisdiction.VIETNAM,
            variables=["research_question", "legal_area", "jurisdiction", "specific_focus", "additional_context", "research_scope"],
            examples=[
                {
                    "input": "Research on data protection laws in Vietnam",
                    "output": "Comprehensive research with current regulations and compliance requirements"
                }
            ]
        )
        
        # Contract Review Template
        self.templates["contract_review"] = PromptTemplate(
            name="Contract Review and Analysis",
            task_type=LegalTaskType.CONTRACT_REVIEW,
            system_prompt="""You are an expert contract lawyer specializing in Vietnamese commercial law and international contract law. Your role is to provide thorough contract review and risk assessment.

Review Methodology:
- Analyze contract structure and completeness
- Identify potential legal risks and liabilities
- Assess enforceability and compliance issues
- Review terms for fairness and balance
- Check for missing or problematic clauses
- Ensure compliance with applicable laws

Contract Review Framework:
1. Contract Overview and Structure
2. Party Identification and Capacity
3. Key Terms and Conditions Analysis
4. Risk Assessment and Liability Issues
5. Compliance and Regulatory Considerations
6. Enforceability and Dispute Resolution
7. Missing or Recommended Clauses
8. Overall Assessment and Recommendations""",
            user_prompt_template="""Please review the following contract:

**Contract Type:** {contract_type}
**Parties:** {parties}
**Jurisdiction:** {jurisdiction}
**Review Purpose:** {review_purpose}

**Contract Content:**
{contract_content}

**Specific Review Focus:**
{review_focus}

**Risk Tolerance:** {risk_tolerance}

Please provide a comprehensive contract review following the framework, highlighting key risks and providing actionable recommendations.""",
            document_type=DocumentType.CONTRACT,
            variables=["contract_type", "parties", "jurisdiction", "review_purpose", "contract_content", "review_focus", "risk_tolerance"],
            examples=[
                {
                    "input": "Software licensing agreement review",
                    "output": "Detailed risk assessment with recommended modifications"
                }
            ]
        )
        
        # Compliance Check Template
        self.templates["compliance_check"] = PromptTemplate(
            name="Legal Compliance Assessment",
            task_type=LegalTaskType.COMPLIANCE_CHECK,
            system_prompt="""You are a compliance specialist with expertise in Vietnamese regulatory frameworks and international compliance standards. Your role is to assess compliance with applicable laws and regulations.

Compliance Assessment Framework:
- Identify applicable laws and regulations
- Assess current compliance status
- Identify gaps and non-compliance risks
- Provide remediation recommendations
- Consider enforcement trends and regulatory updates

Assessment Structure:
1. Regulatory Landscape Overview
2. Applicable Laws and Regulations
3. Current Compliance Status
4. Gap Analysis and Risk Assessment
5. Non-Compliance Consequences
6. Remediation Plan and Recommendations
7. Ongoing Compliance Monitoring
8. Best Practices and Industry Standards""",
            user_prompt_template="""Please conduct a compliance assessment for:

**Organization/Activity:** {organization}
**Industry/Sector:** {industry}
**Jurisdiction:** {jurisdiction}
**Compliance Area:** {compliance_area}

**Current Practices/Policies:**
{current_practices}

**Specific Compliance Questions:**
{compliance_questions}

**Risk Tolerance:** {risk_tolerance}

Please provide a comprehensive compliance assessment following the framework, with specific recommendations for achieving and maintaining compliance.""",
            jurisdiction=LegalJurisdiction.VIETNAM,
            variables=["organization", "industry", "jurisdiction", "compliance_area", "current_practices", "compliance_questions", "risk_tolerance"],
            examples=[
                {
                    "input": "GDPR compliance for Vietnamese company",
                    "output": "Detailed compliance gap analysis with implementation roadmap"
                }
            ]
        )
        
        # Legal Q&A Template
        self.templates["legal_qa"] = PromptTemplate(
            name="Legal Question Answering",
            task_type=LegalTaskType.LEGAL_QA,
            system_prompt="""You are a knowledgeable legal advisor specializing in Vietnamese law with broad expertise across multiple legal domains. Your role is to provide clear, accurate, and practical legal guidance.

Response Guidelines:
- Provide clear, understandable explanations
- Reference relevant laws and regulations
- Distinguish between legal requirements and best practices
- Highlight when professional legal consultation is recommended
- Consider practical implications and real-world applications
- Maintain appropriate disclaimers about legal advice

Response Structure:
1. Direct Answer to the Question
2. Legal Basis and Relevant Laws
3. Practical Implications
4. Potential Risks or Considerations
5. Recommended Actions or Next Steps
6. When to Seek Professional Legal Advice""",
            user_prompt_template="""Please answer the following legal question:

**Question:** {question}
**Legal Area:** {legal_area}
**Jurisdiction:** {jurisdiction}
**Context:** {context}

**Questioner Background:** {questioner_background}
**Urgency Level:** {urgency_level}

Please provide a comprehensive answer following the response structure, ensuring clarity and practical utility.""",
            jurisdiction=LegalJurisdiction.VIETNAM,
            variables=["question", "legal_area", "jurisdiction", "context", "questioner_background", "urgency_level"],
            examples=[
                {
                    "input": "Employment termination procedures in Vietnam",
                    "output": "Step-by-step guidance with legal requirements and best practices"
                }
            ]
        )
        
        # Case Analysis Template
        self.templates["case_analysis"] = PromptTemplate(
            name="Legal Case Analysis",
            task_type=LegalTaskType.CASE_ANALYSIS,
            system_prompt="""You are an expert legal analyst specializing in case law analysis and legal precedent research. Your role is to provide thorough analysis of legal cases and their implications.

Case Analysis Framework:
- Identify key legal issues and holdings
- Analyze reasoning and legal principles
- Assess precedential value and scope
- Consider practical implications
- Compare with related cases and legal developments

Analysis Structure:
1. Case Overview and Procedural History
2. Key Facts and Legal Issues
3. Court's Holding and Reasoning
4. Legal Principles and Precedents Applied
5. Significance and Precedential Value
6. Practical Implications and Applications
7. Related Cases and Legal Developments
8. Critical Analysis and Commentary""",
            user_prompt_template="""Please analyze the following legal case:

**Case Name:** {case_name}
**Court:** {court}
**Date:** {date}
**Legal Area:** {legal_area}

**Case Facts:**
{case_facts}

**Legal Issues:**
{legal_issues}

**Court Decision:**
{court_decision}

**Analysis Focus:** {analysis_focus}

Please provide comprehensive case analysis following the framework, highlighting key legal principles and practical implications.""",
            document_type=DocumentType.CASE_LAW,
            variables=["case_name", "court", "date", "legal_area", "case_facts", "legal_issues", "court_decision", "analysis_focus"],
            examples=[
                {
                    "input": "Supreme Court decision on contract interpretation",
                    "output": "Detailed analysis with precedential implications"
                }
            ]
        )
        
        # Risk Assessment Template
        self.templates["risk_assessment"] = PromptTemplate(
            name="Legal Risk Assessment",
            task_type=LegalTaskType.RISK_ASSESSMENT,
            system_prompt="""You are a legal risk management specialist with expertise in identifying, analyzing, and mitigating legal risks across various business and legal contexts.

Risk Assessment Methodology:
- Identify potential legal risks and exposures
- Assess probability and impact of risks
- Analyze regulatory and compliance risks
- Evaluate litigation and enforcement risks
- Provide risk mitigation strategies

Risk Assessment Framework:
1. Risk Identification and Categorization
2. Probability and Impact Analysis
3. Regulatory and Compliance Risks
4. Litigation and Enforcement Exposure
5. Financial and Operational Impact
6. Risk Mitigation Strategies
7. Monitoring and Review Procedures
8. Risk Management Recommendations""",
            user_prompt_template="""Please conduct a legal risk assessment for:

**Subject:** {subject}
**Business Context:** {business_context}
**Jurisdiction:** {jurisdiction}
**Time Horizon:** {time_horizon}

**Current Situation:**
{current_situation}

**Specific Risk Areas:** {risk_areas}
**Risk Tolerance:** {risk_tolerance}

Please provide a comprehensive risk assessment following the framework, with prioritized recommendations for risk mitigation.""",
            jurisdiction=LegalJurisdiction.VIETNAM,
            variables=["subject", "business_context", "jurisdiction", "time_horizon", "current_situation", "risk_areas", "risk_tolerance"],
            examples=[
                {
                    "input": "Cross-border data transfer risk assessment",
                    "output": "Comprehensive risk analysis with mitigation strategies"
                }
            ]
        )
    
    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """Get a specific prompt template by name"""
        return self.templates.get(template_name)
    
    def get_templates_by_task_type(self, task_type: LegalTaskType) -> List[PromptTemplate]:
        """Get all templates for a specific task type"""
        return [template for template in self.templates.values() 
                if template.task_type == task_type]
    
    def get_templates_by_jurisdiction(self, jurisdiction: LegalJurisdiction) -> List[PromptTemplate]:
        """Get all templates for a specific jurisdiction"""
        return [template for template in self.templates.values() 
                if template.jurisdiction == jurisdiction]
    
    def list_templates(self) -> List[str]:
        """List all available template names"""
        return list(self.templates.keys())
    
    def add_custom_template(self, template: PromptTemplate):
        """Add a custom prompt template"""
        self.templates[template.name.lower().replace(" ", "_")] = template
    
    def format_template(self, template_name: str, **kwargs) -> Dict[str, str]:
        """Format a specific template with provided variables"""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        return template.format_prompt(**kwargs)


class PromptTemplateManager:
    """Manager for prompt templates with advanced features"""
    
    def __init__(self):
        self.templates = LegalPromptTemplates()
        self.usage_history: List[Dict[str, Any]] = []
        self.custom_templates: Dict[str, PromptTemplate] = {}
    
    def get_template_suggestions(
        self, 
        task_type: Optional[LegalTaskType] = None,
        jurisdiction: Optional[LegalJurisdiction] = None,
        document_type: Optional[DocumentType] = None
    ) -> List[str]:
        """Get template suggestions based on criteria"""
        suggestions = []
        
        for name, template in self.templates.templates.items():
            if task_type and template.task_type != task_type:
                continue
            if jurisdiction and template.jurisdiction != jurisdiction:
                continue
            if document_type and template.document_type != document_type:
                continue
            suggestions.append(name)
        
        return suggestions
    
    def create_prompt(
        self, 
        template_name: str, 
        variables: Dict[str, Any],
        track_usage: bool = True
    ) -> Dict[str, str]:
        """Create a formatted prompt from template"""
        try:
            prompt = self.templates.format_template(template_name, **variables)
            
            if track_usage:
                self.usage_history.append({
                    "template_name": template_name,
                    "timestamp": datetime.now().isoformat(),
                    "variables": list(variables.keys()),
                    "success": True
                })
            
            return prompt
            
        except Exception as e:
            if track_usage:
                self.usage_history.append({
                    "template_name": template_name,
                    "timestamp": datetime.now().isoformat(),
                    "variables": list(variables.keys()) if variables else [],
                    "success": False,
                    "error": str(e)
                })
            raise
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for templates"""
        if not self.usage_history:
            return {"total_uses": 0, "success_rate": 0, "popular_templates": []}
        
        total_uses = len(self.usage_history)
        successful_uses = sum(1 for entry in self.usage_history if entry.get("success", False))
        success_rate = successful_uses / total_uses if total_uses > 0 else 0
        
        # Count template usage
        template_counts = {}
        for entry in self.usage_history:
            template_name = entry["template_name"]
            template_counts[template_name] = template_counts.get(template_name, 0) + 1
        
        popular_templates = sorted(template_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_uses": total_uses,
            "success_rate": success_rate,
            "popular_templates": popular_templates[:5],
            "recent_activity": self.usage_history[-10:]
        }
    
    def validate_template_variables(self, template_name: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Validate variables for a template"""
        template = self.templates.get_template(template_name)
        if not template:
            return {"valid": False, "error": f"Template '{template_name}' not found"}
        
        missing_vars = [var for var in template.variables if var not in variables]
        extra_vars = [var for var in variables if var not in template.variables]
        
        return {
            "valid": len(missing_vars) == 0,
            "missing_variables": missing_vars,
            "extra_variables": extra_vars,
            "required_variables": template.variables
        }


# Factory functions for easy usage
def create_prompt_manager() -> PromptTemplateManager:
    """Create a new prompt template manager"""
    return PromptTemplateManager()


def get_legal_prompt(
    task_type: LegalTaskType,
    jurisdiction: LegalJurisdiction = LegalJurisdiction.VIETNAM,
    **variables
) -> Dict[str, str]:
    """Quick function to get a legal prompt"""
    manager = create_prompt_manager()
    suggestions = manager.get_template_suggestions(task_type=task_type, jurisdiction=jurisdiction)
    
    if not suggestions:
        raise ValueError(f"No templates found for task type {task_type} and jurisdiction {jurisdiction}")
    
    template_name = suggestions[0]  # Use first suggestion
    return manager.create_prompt(template_name, variables)


# Example usage
if __name__ == "__main__":
    # Example: Document analysis
    manager = create_prompt_manager()
    
    prompt = manager.create_prompt(
        "document_analysis",
        {
            "document_type": "Employment Contract",
            "document_title": "Software Developer Employment Agreement",
            "jurisdiction": "Vietnam",
            "document_content": "Sample contract content...",
            "analysis_requirements": "Focus on termination clauses and intellectual property rights"
        }
    )
    
    print("System Prompt:", prompt["system"])
    print("\nUser Prompt:", prompt["user"])
    
    # Example: Legal research
    research_prompt = manager.create_prompt(
        "legal_research",
        {
            "research_question": "What are the requirements for data localization in Vietnam?",
            "legal_area": "Data Protection and Privacy",
            "jurisdiction": "Vietnam",
            "specific_focus": "Cross-border data transfer restrictions",
            "additional_context": "For a fintech company processing customer data",
            "research_scope": "Current regulations and upcoming changes"
        }
    )
    
    print("\n" + "="*50)
    print("Research Prompt:", research_prompt["user"])
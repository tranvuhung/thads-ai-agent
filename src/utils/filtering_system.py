"""
Advanced Filtering System for Semantic Search Results

This module provides sophisticated filtering capabilities for search results
based on multiple criteria including content, metadata, temporal, and custom filters.
"""

import logging
import re
import math
from typing import List, Dict, Any, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class FilterType(Enum):
    """Types of filters available"""
    CONTENT_FILTER = "content_filter"
    METADATA_FILTER = "metadata_filter"
    TEMPORAL_FILTER = "temporal_filter"
    SIMILARITY_FILTER = "similarity_filter"
    QUALITY_FILTER = "quality_filter"
    AUTHORITY_FILTER = "authority_filter"
    LANGUAGE_FILTER = "language_filter"
    CUSTOM_FILTER = "custom_filter"


class FilterOperator(Enum):
    """Filter operators"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX_MATCH = "regex_match"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    BETWEEN = "between"
    IN_LIST = "in_list"
    NOT_IN_LIST = "not_in_list"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class LogicalOperator(Enum):
    """Logical operators for combining filters"""
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class FilterCriterion:
    """Single filter criterion"""
    field: str
    operator: FilterOperator
    value: Any
    filter_type: FilterType = FilterType.METADATA_FILTER
    case_sensitive: bool = False
    weight: float = 1.0
    description: str = ""


@dataclass
class FilterGroup:
    """Group of filter criteria with logical operator"""
    criteria: List[Union[FilterCriterion, 'FilterGroup']]
    logical_operator: LogicalOperator = LogicalOperator.AND
    weight: float = 1.0
    description: str = ""


@dataclass
class FilterConfig:
    """Configuration for filtering system"""
    enable_fuzzy_matching: bool = True
    fuzzy_threshold: float = 0.8
    enable_stemming: bool = True
    enable_synonyms: bool = True
    case_sensitive_default: bool = False
    max_results: Optional[int] = None
    min_score_threshold: float = 0.0
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds


@dataclass
class FilterResult:
    """Result of filtering operation"""
    filtered_results: List[Any]
    filter_stats: Dict[str, Any]
    applied_filters: List[FilterCriterion]
    execution_time: float
    cache_hit: bool = False


class AdvancedFilteringSystem:
    """
    Advanced filtering system with multiple criteria and operators
    """
    
    def __init__(self, config: FilterConfig = None):
        """
        Initialize the filtering system
        
        Args:
            config: Filtering configuration
        """
        self.config = config or FilterConfig()
        self.filter_cache = {}
        self.synonyms = self._load_synonyms()
        self.stemmer = self._initialize_stemmer()
        
        logger.info("AdvancedFilteringSystem initialized")
    
    def apply_filters(self, 
                     results: List[Any],
                     filters: Union[FilterCriterion, FilterGroup, List[FilterCriterion]],
                     query_context: Dict[str, Any] = None) -> FilterResult:
        """
        Apply filters to search results
        
        Args:
            results: Search results to filter
            filters: Filter criteria or group
            query_context: Additional context for filtering
            
        Returns:
            FilterResult with filtered results and statistics
        """
        start_time = datetime.now()
        
        # Normalize filters to FilterGroup
        if isinstance(filters, FilterCriterion):
            filter_group = FilterGroup(criteria=[filters])
        elif isinstance(filters, list):
            filter_group = FilterGroup(criteria=filters)
        else:
            filter_group = filters
        
        # Check cache
        cache_key = self._generate_cache_key(results, filter_group)
        if self.config.enable_caching and cache_key in self.filter_cache:
            cached_result = self.filter_cache[cache_key]
            if self._is_cache_valid(cached_result):
                cached_result.cache_hit = True
                return cached_result
        
        # Apply filters
        filtered_results = self._apply_filter_group(results, filter_group, query_context)
        
        # Apply global constraints
        if self.config.min_score_threshold > 0:
            filtered_results = self._apply_score_threshold(filtered_results)
        
        if self.config.max_results:
            filtered_results = filtered_results[:self.config.max_results]
        
        # Calculate statistics
        execution_time = (datetime.now() - start_time).total_seconds()
        filter_stats = self._calculate_filter_stats(results, filtered_results, filter_group)
        
        # Create result
        result = FilterResult(
            filtered_results=filtered_results,
            filter_stats=filter_stats,
            applied_filters=self._extract_filter_criteria(filter_group),
            execution_time=execution_time,
            cache_hit=False
        )
        
        # Cache result
        if self.config.enable_caching:
            self.filter_cache[cache_key] = result
        
        return result
    
    def _apply_filter_group(self, 
                           results: List[Any], 
                           filter_group: FilterGroup,
                           query_context: Dict[str, Any] = None) -> List[Any]:
        """Apply a group of filters with logical operators"""
        if not filter_group.criteria:
            return results
        
        if filter_group.logical_operator == LogicalOperator.AND:
            filtered_results = results
            for criterion in filter_group.criteria:
                if isinstance(criterion, FilterCriterion):
                    filtered_results = self._apply_single_filter(filtered_results, criterion, query_context)
                elif isinstance(criterion, FilterGroup):
                    filtered_results = self._apply_filter_group(filtered_results, criterion, query_context)
            return filtered_results
        
        elif filter_group.logical_operator == LogicalOperator.OR:
            all_matches = set()
            for criterion in filter_group.criteria:
                if isinstance(criterion, FilterCriterion):
                    matches = self._apply_single_filter(results, criterion, query_context)
                elif isinstance(criterion, FilterGroup):
                    matches = self._apply_filter_group(results, criterion, query_context)
                
                # Add result IDs to set
                for result in matches:
                    result_id = self._get_result_id(result)
                    all_matches.add(result_id)
            
            # Return results that match any criterion
            return [r for r in results if self._get_result_id(r) in all_matches]
        
        elif filter_group.logical_operator == LogicalOperator.NOT:
            # Apply first criterion and return inverse
            if filter_group.criteria:
                first_criterion = filter_group.criteria[0]
                if isinstance(first_criterion, FilterCriterion):
                    matches = self._apply_single_filter(results, first_criterion, query_context)
                elif isinstance(first_criterion, FilterGroup):
                    matches = self._apply_filter_group(results, first_criterion, query_context)
                
                match_ids = {self._get_result_id(r) for r in matches}
                return [r for r in results if self._get_result_id(r) not in match_ids]
        
        return results
    
    def _apply_single_filter(self, 
                           results: List[Any], 
                           criterion: FilterCriterion,
                           query_context: Dict[str, Any] = None) -> List[Any]:
        """Apply a single filter criterion"""
        filtered_results = []
        
        for result in results:
            if self._evaluate_criterion(result, criterion, query_context):
                filtered_results.append(result)
        
        return filtered_results
    
    def _evaluate_criterion(self, 
                          result: Any, 
                          criterion: FilterCriterion,
                          query_context: Dict[str, Any] = None) -> bool:
        """Evaluate if a result matches a filter criterion"""
        try:
            # Extract field value from result
            field_value = self._extract_field_value(result, criterion.field, criterion.filter_type)
            
            # Apply operator
            return self._apply_operator(field_value, criterion.operator, criterion.value, criterion.case_sensitive)
            
        except Exception as e:
            logger.warning(f"Error evaluating criterion {criterion.field}: {e}")
            return False
    
    def _extract_field_value(self, result: Any, field: str, filter_type: FilterType) -> Any:
        """Extract field value from result based on filter type"""
        if filter_type == FilterType.CONTENT_FILTER:
            return getattr(result, 'content', '')
        
        elif filter_type == FilterType.METADATA_FILTER:
            metadata = getattr(result, 'chunk_metadata', {}) or getattr(result, 'metadata', {})
            return self._get_nested_value(metadata, field)
        
        elif filter_type == FilterType.SIMILARITY_FILTER:
            if field == 'similarity_score':
                return getattr(result, 'similarity_score', 0.0)
            elif field == 'ranking_score':
                return getattr(result, 'ranking_score', 0.0)
        
        elif filter_type == FilterType.TEMPORAL_FILTER:
            metadata = getattr(result, 'chunk_metadata', {}) or getattr(result, 'metadata', {})
            date_field = metadata.get(field) or metadata.get('created_date') or metadata.get('date')
            if isinstance(date_field, str):
                try:
                    return datetime.fromisoformat(date_field.replace('Z', '+00:00'))
                except:
                    return None
            return date_field
        
        elif filter_type == FilterType.QUALITY_FILTER:
            return self._calculate_quality_score(result)
        
        elif filter_type == FilterType.AUTHORITY_FILTER:
            return self._calculate_authority_score(result)
        
        elif filter_type == FilterType.LANGUAGE_FILTER:
            metadata = getattr(result, 'chunk_metadata', {}) or getattr(result, 'metadata', {})
            return metadata.get('language', self._detect_language(getattr(result, 'content', '')))
        
        else:
            # Try to get field directly from result
            return getattr(result, field, None)
    
    def _get_nested_value(self, data: Dict[str, Any], field: str) -> Any:
        """Get nested value from dictionary using dot notation"""
        keys = field.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _apply_operator(self, field_value: Any, operator: FilterOperator, filter_value: Any, case_sensitive: bool = False) -> bool:
        """Apply filter operator to field value"""
        # Handle None values
        if operator == FilterOperator.IS_NULL:
            return field_value is None
        elif operator == FilterOperator.IS_NOT_NULL:
            return field_value is not None
        elif field_value is None:
            return False
        
        # String operations
        if isinstance(field_value, str):
            if not case_sensitive:
                field_value = field_value.lower()
                if isinstance(filter_value, str):
                    filter_value = filter_value.lower()
            
            if operator == FilterOperator.EQUALS:
                return field_value == filter_value
            elif operator == FilterOperator.NOT_EQUALS:
                return field_value != filter_value
            elif operator == FilterOperator.CONTAINS:
                return str(filter_value) in field_value
            elif operator == FilterOperator.NOT_CONTAINS:
                return str(filter_value) not in field_value
            elif operator == FilterOperator.STARTS_WITH:
                return field_value.startswith(str(filter_value))
            elif operator == FilterOperator.ENDS_WITH:
                return field_value.endswith(str(filter_value))
            elif operator == FilterOperator.REGEX_MATCH:
                try:
                    pattern = re.compile(str(filter_value), re.IGNORECASE if not case_sensitive else 0)
                    return bool(pattern.search(field_value))
                except re.error:
                    return False
        
        # Numeric operations
        if isinstance(field_value, (int, float)):
            try:
                filter_value = float(filter_value)
                
                if operator == FilterOperator.EQUALS:
                    return abs(field_value - filter_value) < 1e-9
                elif operator == FilterOperator.NOT_EQUALS:
                    return abs(field_value - filter_value) >= 1e-9
                elif operator == FilterOperator.GREATER_THAN:
                    return field_value > filter_value
                elif operator == FilterOperator.LESS_THAN:
                    return field_value < filter_value
                elif operator == FilterOperator.GREATER_EQUAL:
                    return field_value >= filter_value
                elif operator == FilterOperator.LESS_EQUAL:
                    return field_value <= filter_value
                elif operator == FilterOperator.BETWEEN:
                    if isinstance(filter_value, (list, tuple)) and len(filter_value) == 2:
                        return filter_value[0] <= field_value <= filter_value[1]
            except (ValueError, TypeError):
                pass
        
        # List operations
        if operator == FilterOperator.IN_LIST:
            if isinstance(filter_value, (list, tuple, set)):
                return field_value in filter_value
            else:
                return field_value == filter_value
        elif operator == FilterOperator.NOT_IN_LIST:
            if isinstance(filter_value, (list, tuple, set)):
                return field_value not in filter_value
            else:
                return field_value != filter_value
        
        # Date operations
        if isinstance(field_value, datetime):
            try:
                if isinstance(filter_value, str):
                    filter_value = datetime.fromisoformat(filter_value.replace('Z', '+00:00'))
                elif isinstance(filter_value, (list, tuple)) and len(filter_value) == 2:
                    # For BETWEEN operator with dates
                    start_date = datetime.fromisoformat(filter_value[0].replace('Z', '+00:00')) if isinstance(filter_value[0], str) else filter_value[0]
                    end_date = datetime.fromisoformat(filter_value[1].replace('Z', '+00:00')) if isinstance(filter_value[1], str) else filter_value[1]
                    filter_value = (start_date, end_date)
                
                if operator == FilterOperator.EQUALS:
                    return field_value.date() == filter_value.date()
                elif operator == FilterOperator.GREATER_THAN:
                    return field_value > filter_value
                elif operator == FilterOperator.LESS_THAN:
                    return field_value < filter_value
                elif operator == FilterOperator.GREATER_EQUAL:
                    return field_value >= filter_value
                elif operator == FilterOperator.LESS_EQUAL:
                    return field_value <= filter_value
                elif operator == FilterOperator.BETWEEN:
                    if isinstance(filter_value, (list, tuple)) and len(filter_value) == 2:
                        return filter_value[0] <= field_value <= filter_value[1]
            except (ValueError, TypeError):
                pass
        
        # Default equality check
        return field_value == filter_value
    
    def _apply_score_threshold(self, results: List[Any]) -> List[Any]:
        """Apply minimum score threshold"""
        filtered_results = []
        
        for result in results:
            score = getattr(result, 'similarity_score', 0.0) or getattr(result, 'ranking_score', 0.0)
            if score >= self.config.min_score_threshold:
                filtered_results.append(result)
        
        return filtered_results
    
    def _calculate_quality_score(self, result: Any) -> float:
        """Calculate quality score for a result"""
        content = getattr(result, 'content', '')
        metadata = getattr(result, 'chunk_metadata', {}) or getattr(result, 'metadata', {})
        
        quality_score = 0.5  # Base score
        
        # Content length quality
        length = len(content)
        if 50 <= length <= 2000:
            quality_score += 0.2
        elif length < 50:
            quality_score -= 0.2
        
        # Language quality (basic checks)
        sentences = content.split('.')
        if len(sentences) > 1:
            quality_score += 0.1
        
        # Metadata completeness
        if metadata.get('title'):
            quality_score += 0.1
        if metadata.get('author'):
            quality_score += 0.1
        
        # Formatting quality
        if any(char in content for char in ['\n', '\t']):
            quality_score += 0.1
        
        return min(1.0, max(0.0, quality_score))
    
    def _calculate_authority_score(self, result: Any) -> float:
        """Calculate authority score for a result"""
        metadata = getattr(result, 'chunk_metadata', {}) or getattr(result, 'metadata', {})
        
        authority_score = 0.5  # Base score
        
        # Document type authority
        doc_type = metadata.get('document_type', '').lower()
        if 'law' in doc_type or 'regulation' in doc_type:
            authority_score += 0.3
        elif 'official' in doc_type:
            authority_score += 0.2
        
        # Source authority
        source = metadata.get('source', '').lower()
        if 'government' in source or 'official' in source:
            authority_score += 0.2
        
        # Citation count
        citations = metadata.get('citation_count', 0)
        if citations > 0:
            authority_score += min(0.3, math.log(citations + 1) / 10)
        
        return min(1.0, max(0.0, authority_score))
    
    def _detect_language(self, content: str) -> str:
        """Simple language detection"""
        # This is a very basic implementation
        # In practice, you'd use a proper language detection library
        
        vietnamese_chars = 'àáâãèéêìíòóôõùúăđĩũơưăạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ'
        
        if any(char in content.lower() for char in vietnamese_chars):
            return 'vi'
        else:
            return 'en'
    
    def _get_result_id(self, result: Any) -> str:
        """Get unique identifier for a result"""
        return getattr(result, 'chunk_id', '') or getattr(result, 'id', '') or str(hash(str(result)))
    
    def _calculate_filter_stats(self, 
                              original_results: List[Any], 
                              filtered_results: List[Any],
                              filter_group: FilterGroup) -> Dict[str, Any]:
        """Calculate filtering statistics"""
        return {
            'original_count': len(original_results),
            'filtered_count': len(filtered_results),
            'filter_ratio': len(filtered_results) / len(original_results) if original_results else 0.0,
            'filters_applied': len(self._extract_filter_criteria(filter_group)),
            'filter_types': list(set(c.filter_type.value for c in self._extract_filter_criteria(filter_group)))
        }
    
    def _extract_filter_criteria(self, filter_group: FilterGroup) -> List[FilterCriterion]:
        """Extract all filter criteria from a filter group"""
        criteria = []
        
        for item in filter_group.criteria:
            if isinstance(item, FilterCriterion):
                criteria.append(item)
            elif isinstance(item, FilterGroup):
                criteria.extend(self._extract_filter_criteria(item))
        
        return criteria
    
    def _generate_cache_key(self, results: List[Any], filter_group: FilterGroup) -> str:
        """Generate cache key for filter operation"""
        result_hash = hash(tuple(self._get_result_id(r) for r in results))
        filter_hash = hash(str(filter_group))
        return f"{result_hash}_{filter_hash}"
    
    def _is_cache_valid(self, cached_result: FilterResult) -> bool:
        """Check if cached result is still valid"""
        if not self.config.enable_caching:
            return False
        
        # Simple TTL check (in practice, you might want more sophisticated cache invalidation)
        return True  # For now, assume cache is always valid
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Load synonym dictionary"""
        # This would load from a file or database in practice
        return {
            'law': ['regulation', 'statute', 'act', 'code'],
            'document': ['file', 'text', 'paper', 'record'],
            'search': ['find', 'lookup', 'query', 'retrieve']
        }
    
    def _initialize_stemmer(self):
        """Initialize stemmer for text processing"""
        # This would initialize a proper stemmer in practice
        return None
    
    def create_content_filter(self, 
                            query: str, 
                            operator: FilterOperator = FilterOperator.CONTAINS,
                            case_sensitive: bool = False) -> FilterCriterion:
        """Create a content-based filter"""
        return FilterCriterion(
            field='content',
            operator=operator,
            value=query,
            filter_type=FilterType.CONTENT_FILTER,
            case_sensitive=case_sensitive,
            description=f"Content {operator.value} '{query}'"
        )
    
    def create_metadata_filter(self, 
                             field: str, 
                             value: Any,
                             operator: FilterOperator = FilterOperator.EQUALS) -> FilterCriterion:
        """Create a metadata-based filter"""
        return FilterCriterion(
            field=field,
            operator=operator,
            value=value,
            filter_type=FilterType.METADATA_FILTER,
            description=f"Metadata {field} {operator.value} {value}"
        )
    
    def create_temporal_filter(self, 
                             start_date: datetime = None,
                             end_date: datetime = None,
                             field: str = 'created_date') -> FilterCriterion:
        """Create a temporal filter"""
        if start_date and end_date:
            return FilterCriterion(
                field=field,
                operator=FilterOperator.BETWEEN,
                value=(start_date, end_date),
                filter_type=FilterType.TEMPORAL_FILTER,
                description=f"Date between {start_date} and {end_date}"
            )
        elif start_date:
            return FilterCriterion(
                field=field,
                operator=FilterOperator.GREATER_EQUAL,
                value=start_date,
                filter_type=FilterType.TEMPORAL_FILTER,
                description=f"Date after {start_date}"
            )
        elif end_date:
            return FilterCriterion(
                field=field,
                operator=FilterOperator.LESS_EQUAL,
                value=end_date,
                filter_type=FilterType.TEMPORAL_FILTER,
                description=f"Date before {end_date}"
            )
    
    def create_similarity_filter(self, 
                               min_score: float,
                               score_field: str = 'similarity_score') -> FilterCriterion:
        """Create a similarity score filter"""
        return FilterCriterion(
            field=score_field,
            operator=FilterOperator.GREATER_EQUAL,
            value=min_score,
            filter_type=FilterType.SIMILARITY_FILTER,
            description=f"Similarity score >= {min_score}"
        )
    
    def create_quality_filter(self, min_quality: float = 0.5) -> FilterCriterion:
        """Create a quality filter"""
        return FilterCriterion(
            field='quality_score',
            operator=FilterOperator.GREATER_EQUAL,
            value=min_quality,
            filter_type=FilterType.QUALITY_FILTER,
            description=f"Quality score >= {min_quality}"
        )
    
    def create_authority_filter(self, min_authority: float = 0.5) -> FilterCriterion:
        """Create an authority filter"""
        return FilterCriterion(
            field='authority_score',
            operator=FilterOperator.GREATER_EQUAL,
            value=min_authority,
            filter_type=FilterType.AUTHORITY_FILTER,
            description=f"Authority score >= {min_authority}"
        )
    
    def create_language_filter(self, languages: Union[str, List[str]]) -> FilterCriterion:
        """Create a language filter"""
        if isinstance(languages, str):
            languages = [languages]
        
        return FilterCriterion(
            field='language',
            operator=FilterOperator.IN_LIST,
            value=languages,
            filter_type=FilterType.LANGUAGE_FILTER,
            description=f"Language in {languages}"
        )
    
    def clear_cache(self):
        """Clear the filter cache"""
        self.filter_cache.clear()
        logger.info("Filter cache cleared")
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get filtering system statistics"""
        return {
            'cache_size': len(self.filter_cache),
            'cache_enabled': self.config.enable_caching,
            'fuzzy_matching': self.config.enable_fuzzy_matching,
            'stemming_enabled': self.config.enable_stemming,
            'synonyms_loaded': len(self.synonyms),
            'max_results': self.config.max_results,
            'min_score_threshold': self.config.min_score_threshold
        }


# Utility functions

def create_advanced_filter_group(filters: List[FilterCriterion], 
                               operator: LogicalOperator = LogicalOperator.AND) -> FilterGroup:
    """Create an advanced filter group with multiple criteria"""
    return FilterGroup(
        criteria=filters,
        logical_operator=operator,
        description=f"Group with {len(filters)} filters using {operator.value}"
    )


def create_date_range_filter(start_date: str, 
                           end_date: str,
                           field: str = 'created_date') -> FilterCriterion:
    """Create a date range filter from string dates"""
    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
    
    return FilterCriterion(
        field=field,
        operator=FilterOperator.BETWEEN,
        value=(start_dt, end_dt),
        filter_type=FilterType.TEMPORAL_FILTER,
        description=f"Date range {start_date} to {end_date}"
    )


def create_multi_field_search_filter(query: str, 
                                   fields: List[str],
                                   operator: FilterOperator = FilterOperator.CONTAINS) -> FilterGroup:
    """Create a filter that searches across multiple fields"""
    criteria = []
    
    for field in fields:
        criterion = FilterCriterion(
            field=field,
            operator=operator,
            value=query,
            filter_type=FilterType.METADATA_FILTER,
            description=f"Search '{query}' in {field}"
        )
        criteria.append(criterion)
    
    return FilterGroup(
        criteria=criteria,
        logical_operator=LogicalOperator.OR,
        description=f"Multi-field search for '{query}'"
    )


def validate_filter_criterion(criterion: FilterCriterion) -> bool:
    """Validate a filter criterion"""
    if not criterion.field:
        return False
    
    if criterion.operator == FilterOperator.BETWEEN:
        if not isinstance(criterion.value, (list, tuple)) or len(criterion.value) != 2:
            return False
    
    if criterion.operator in [FilterOperator.IN_LIST, FilterOperator.NOT_IN_LIST]:
        if not isinstance(criterion.value, (list, tuple, set)):
            return False
    
    return True
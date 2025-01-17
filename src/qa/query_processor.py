from typing import Dict, List, Tuple, Optional, Any
import re
from enum import Enum

class QueryType(Enum):
    NUTRITIONAL = "nutritional"  # e.g., "How much protein in..."
    COMPARISON = "comparison"    # e.g., "Compare ... and ..."
    RECOMMENDATION = "recommendation"  # e.g., "What foods are high in..."
    GENERAL = "general"         # e.g., "Tell me about..."
    UNKNOWN = "unknown"

class QueryProcessor:
    def __init__(self):
        # Define patterns for different query types
        self.patterns = {
            QueryType.NUTRITIONAL: [
                r'how much (\w+)',
                r'what is the (\w+) content',
                r'(\w+) content in',
                r'amount of (\w+)',
                r'group (\d+)',
                r'food group (\d+)',
            ],
            QueryType.COMPARISON: [
                r'compare|versus|vs\.',
                r'difference between',
                r'which (has|contains) more',
                r'better source of',
                r'compare group',
                r'which group',
            ],
            QueryType.RECOMMENDATION: [
                r'what foods? (are|is) (high|rich|low) in',
                r'foods? with (high|low)',
                r'best sources? of',
                r'recommend|suggest',
                r'which group (has|contains)',
            ],
            QueryType.GENERAL: [
                r'tell me about',
                r'what is',
                r'describe',
                r'information about',
                r'group information',
            ]
        }
        
        # Define nutritional aspects and their variations
        self.nutritional_aspects = {
            'protein': ['protein', 'proteins', 'amino acids'],
            'carbohydrates': ['carbs', 'carbohydrates', 'sugars', 'sugar'],
            'fat': ['fat', 'fats', 'fatty acids', 'oils'],
            'calories': ['calories', 'caloric', 'energy'],
            'fiber': ['fiber', 'fibre', 'dietary fiber'],
            'vitamin_c': ['vitamin c', 'vit c', 'vitamin-c', 'ascorbic acid'],
        }
        
        # Compile regex patterns
        self.compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict:
        """Compile regex patterns for efficient matching."""
        compiled = {}
        for qtype, patterns in self.patterns.items():
            compiled[qtype] = [re.compile(p, re.IGNORECASE) for p in patterns]
        return compiled

    def classify_query(self, query: str) -> Tuple[QueryType, Dict[str, Any]]:
        """Classify the query type and extract relevant information."""
        query = query.lower().strip()
        
        # Try to match query type
        for qtype, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    # Extract additional information based on query type
                    info = self._extract_query_info(query, qtype)
                    return qtype, info
        
        return QueryType.UNKNOWN, {}

    def _extract_query_info(self, query: str, query_type: QueryType) -> Dict[str, Any]:
        """Extract relevant information from the query based on its type."""
        info = {
            'original_query': query,
            'nutritional_aspects': self._find_nutritional_aspects(query),
            'comparison_items': [],
            'quantity_indicators': self._find_quantity_indicators(query),
        }
        
        if query_type == QueryType.COMPARISON:
            info['comparison_items'] = self._extract_comparison_items(query)
            
        return info

    def _find_nutritional_aspects(self, query: str) -> List[str]:
        """Find mentioned nutritional aspects in the query."""
        found_aspects = []
        for aspect, variations in self.nutritional_aspects.items():
            if any(var in query for var in variations):
                found_aspects.append(aspect)
        return found_aspects

    def _find_quantity_indicators(self, query: str) -> Dict[str, str]:
        """Find quantity indicators (high, low, etc.) in the query."""
        indicators = {
            'high': ['high', 'rich', 'lots', 'plenty', 'abundant'],
            'low': ['low', 'little', 'few', 'minimal'],
            'moderate': ['moderate', 'medium', 'average']
        }
        
        found = {}
        for level, words in indicators.items():
            if any(word in query for word in words):
                found['level'] = level
                break
                
        return found

    def _extract_comparison_items(self, query: str) -> List[str]:
        """Extract items being compared in a comparison query."""
        # Look for patterns like "X vs Y" or "compare X and Y"
        comparison_patterns = [
            r'compare\s+(\w+)\s+and\s+(\w+)',
            r'(\w+)\s+vs\.?\s+(\w+)',
            r'(\w+)\s+versus\s+(\w+)',
            r'difference\s+between\s+(\w+)\s+and\s+(\w+)'
        ]
        
        for pattern in comparison_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return [match.group(1), match.group(2)]
        
        return []

    def generate_search_queries(self, query_type: QueryType, info: Dict) -> List[str]:
        """Generate multiple search queries based on query type and extracted info."""
        queries = []
        
        if query_type == QueryType.NUTRITIONAL:
            for aspect in info['nutritional_aspects']:
                queries.append(f"food {aspect} content")
                if 'level' in info.get('quantity_indicators', {}):
                    level = info['quantity_indicators']['level']
                    queries.append(f"{level} {aspect} food")
                    
        elif query_type == QueryType.COMPARISON:
            if info['comparison_items']:
                items = info['comparison_items']
                queries.extend([
                    f"{item} nutritional content" for item in items
                ])
                if info['nutritional_aspects']:
                    aspect = info['nutritional_aspects'][0]
                    queries.append(f"{items[0]} vs {items[1]} {aspect}")
                    
        elif query_type == QueryType.RECOMMENDATION:
            for aspect in info['nutritional_aspects']:
                level = info.get('quantity_indicators', {}).get('level', 'high')
                queries.append(f"{level} {aspect} foods")
                
        # Always include original query
        queries.append(info['original_query'])
        
        return list(set(queries))  # Remove duplicates 
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ImportWarning)
import torch
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from enum import Enum
import re
from .query_processor import QueryProcessor, QueryType
from .external_fallback import ExternalFallback
import atexit
import torch.multiprocessing as mp
import os

# Configure numpy to use newer API
np.set_printoptions(legacy='1.13')

class AnswerSource(Enum):
    LOCAL = "local"
    EXTERNAL = "external"
    COMBINED = "combined"

class RAGEngine:
    def __init__(self, df: pd.DataFrame, use_external: bool = True):
        """Initialize the RAG engine with a dataframe of food data."""
        # Force CPU usage for better compatibility
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        self.df = df
        self.use_external = use_external
        
        # Initialize model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        self.query_processor = QueryProcessor()
        
        if self.use_external:
            self.external_fallback = ExternalFallback()
        
        # Create food contexts and embeddings
        self.food_contexts = self._create_food_contexts()
        self.embeddings = self._create_embeddings()
        self.index = self._create_index()
        
        # Track data sources
        self.data_sources = set(df['data_source'].unique()) if 'data_source' in df.columns else {'default'}
        
        # Register cleanup handler
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Cleanup resources when the engine is shut down."""
        if hasattr(self, 'model'):
            self.model = None
            
        # Clear other large objects
        if hasattr(self, 'embeddings'):
            self.embeddings = None
        if hasattr(self, 'index'):
            self.index = None
    
    def __del__(self):
        """Destructor to ensure cleanup is called."""
        self._cleanup()

    def _create_food_contexts(self) -> List[str]:
        """Create context strings for each food item."""
        contexts = []
        for _, row in self.df.iterrows():
            # Handle Kaggle dataset format
            if row.get('data_source') == 'kaggle':
                context = f"""
                Food Group: {row.get('Group', 'Unknown')}
                Energy: {row.get('ENERCC (kcal)', 'N/A')} kcal
                Protein: {row.get('PROT (g)', 'N/A')} g
                Carbs: {row.get('CHO (g)', 'N/A')} g
                Fat: {row.get('FAT (g)', 'N/A')} g
                Fiber: {row.get('FIBT (g)', 'N/A')} g
                Source: Kaggle Food Groups Dataset
                """
            # Handle NEVO dataset format
            else:
                context = f"""
                Food: {row['Voedingsmiddelnaam/Dutch food name']}
                Energy: {row['ENERCC (kcal)']} kcal/100g
                Protein: {row['PROT (g)']} g/100g
                Carbs: {row['CHO (g)']} g/100g
                Fat: {row['FAT (g)']} g/100g
                Fiber: {row.get('FIBT (g)', 'N/A')} g/100g
                Vitamin C: {row.get('VITC (mg)', 'N/A')} mg/100g
                Source: NEVO Database
                """
            contexts.append(context.strip())
        return contexts

    def _create_embeddings(self) -> np.ndarray:
        """Create embeddings for all food contexts."""
        return self.model.encode(self.food_contexts, show_progress_bar=True)

    def _create_index(self) -> faiss.IndexFlatL2:
        """Create FAISS index for fast similarity search."""
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(self.embeddings.astype('float32'))
        return index

    def _adjust_ranking(self, query: str, results: List[Tuple[int, float, str]]) -> List[Tuple[int, float, str]]:
        """Adjust ranking based on specific nutritional queries."""
        # Extract nutritional focus from query
        query_lower = query.lower()
        
        # Check for group-specific queries first
        group_match = re.search(r'group\s*(\d+)', query_lower)
        if group_match:
            group_num = group_match.group(1)
            # Prioritize results from the specified group
            sorted_results = []
            for idx, dist, context in results:
                try:
                    data_source = self.df.iloc[idx].get('data_source', '')
                    group = str(self.df.iloc[idx].get('Group', ''))
                    
                    # Calculate adjusted distance:
                    # - Kaggle source with matching group: very low distance (high relevance)
                    # - Kaggle source with non-matching group: medium distance
                    # - NEVO source: high distance (low relevance)
                    if data_source == 'kaggle':
                        if group == group_num:
                            adjusted_dist = float(dist) * 0.01  # Boost matching group results significantly
                        else:
                            adjusted_dist = float(dist) * 1.5   # Slightly penalize non-matching groups
                    else:
                        adjusted_dist = float(dist) * 5.0       # Significantly penalize non-Kaggle results
                    
                    sorted_results.append((idx, adjusted_dist, context))
                except (ValueError, TypeError):
                    sorted_results.append((idx, float(dist) * 5.0, context))
            return sorted(sorted_results, key=lambda x: x[1])
        
        # Handle other nutritional queries
        patterns = {
            'vitamin_c': (r'vitamin\s*c', 'VITC (mg)'),
            'protein': (r'protein', 'PROT (g)'),
            'calories': (r'calorie|energy', 'ENERCC (kcal)'),
            'fiber': (r'fiber|fibre', 'FIBT (g)')
        }
        
        # Check if query matches any nutritional pattern
        for nutrient, (pattern, column) in patterns.items():
            if re.search(pattern, query_lower):
                if column in self.df.columns:
                    sorted_results = []
                    for idx, dist, context in results:
                        try:
                            value = self.df.iloc[idx][column]
                            value = float(value) if pd.notna(value) and str(value).replace('.', '').isdigit() else 0.0
                            dist_float = float(dist)
                            # Also consider data source in ranking
                            source_multiplier = 0.8 if self.df.iloc[idx].get('data_source') == 'kaggle' else 1.0
                            adjusted_dist = (dist_float / (1 + value)) * source_multiplier
                            sorted_results.append((idx, adjusted_dist, context))
                        except (ValueError, TypeError):
                            sorted_results.append((idx, float(dist), context))
                    return sorted(sorted_results, key=lambda x: x[1])
        
        return [(idx, float(dist), context) for idx, dist, context in results]

    def _get_relevant_foods(self, query: str, k: int = 5) -> List[Tuple[int, float, str]]:
        """Get most relevant foods for a query using semantic search."""
        # Create query embedding
        query_embedding = self.model.encode([query])
        
        # Search for similar contexts
        D, I = self.index.search(query_embedding.astype('float32'), k=k)
        
        # Create initial results
        results = [(idx, dist, self.food_contexts[idx]) for idx, dist in zip(I[0], D[0])]
        
        # Adjust ranking based on query content
        return self._adjust_ranking(query, results)

    def _format_answer(self, query: str, relevant_foods: List[Tuple[int, float, str]], query_info: Dict) -> Dict[str, Any]:
        """Format the answer based on relevant foods and query type."""
        if not relevant_foods:
            return {
                "answer": "I couldn't find any relevant food items for your query.",
                "confidence": "low",
                "source": AnswerSource.LOCAL.value
            }

        # Extract food names and their relevance scores
        foods_info = []
        min_relevance_threshold = 0.5  # Increased minimum relevance threshold
        
        # Check if this is a group-specific query
        group_match = re.search(r'group\s*(\d+)', query.lower())
        group_num = group_match.group(1) if group_match else None
        
        for idx, dist, context in relevant_foods:
            source = self.df.iloc[idx].get('data_source', 'unknown')
            relevance = 1 / (1 + float(dist))  # Convert distance to similarity score
            
            # For group queries, only include results from the Kaggle dataset
            if group_num and source != 'kaggle':
                continue
                
            # Only include foods with relevance above threshold
            if relevance >= min_relevance_threshold:
                foods_info.append({
                    "food": f"Food Group {self.df.iloc[idx].get('Group', 'Unknown')}" if source == 'kaggle' 
                           else self.df.iloc[idx]['Voedingsmiddelnaam/Dutch food name'],
                    "relevance": float(relevance),
                    "source": source,
                    "context": context
                })

        # If no foods meet the relevance threshold
        if not foods_info:
            return {
                "answer": "I couldn't find any sufficiently relevant food items for your query.",
                "confidence": "low",
                "source": AnswerSource.LOCAL.value
            }

        # Generate appropriate answer based on query type
        query_type = query_info.get('query_type', QueryType.UNKNOWN)
        
        if group_num:
            answer = f"Here's the nutritional information for Food Group {group_num}:"
        elif query_type == QueryType.COMPARISON and len(query_info.get('comparison_items', [])) == 2:
            answer = "Here's a comparison of " + ' and '.join(query_info['comparison_items']) + ":"
        elif query_type == QueryType.RECOMMENDATION:
            aspects = query_info.get('nutritional_aspects', [])
            level = query_info.get('quantity_indicators', {}).get('level', 'high')
            if aspects:
                answer = f"Here are foods with {level} {aspects[0]} content:"
            else:
                answer = "Here are some recommended foods based on your query:"
        else:
            answer = "Here are the most relevant foods for your query:"

        # Determine confidence based on best match with stricter thresholds
        best_relevance = foods_info[0]["relevance"]
        confidence = "high" if best_relevance > 0.9 else "medium" if best_relevance > 0.7 else "low"

        return {
            "answer": answer,
            "matches": foods_info,
            "confidence": confidence,
            "source": AnswerSource.LOCAL.value,
            "query_understood": query_type != QueryType.UNKNOWN,
            "query_type": query_type.value if hasattr(query_type, 'value') else str(query_type),
            "nutritional_aspects": query_info.get('nutritional_aspects', [])
        }

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a food-related question using RAG approach with external API fallback."""
        try:
            if not question.strip():
                return {
                    "error": "Please provide a non-empty question.",
                    "source": AnswerSource.LOCAL.value,
                    "confidence": "low"
                }
            
            # Process the query
            query_type, query_info = self.query_processor.classify_query(question)
            query_info['query_type'] = query_type
            
            # Get relevant foods using semantic search
            relevant_foods = self._get_relevant_foods(question)
            
            # Format initial response
            response = self._format_answer(question, relevant_foods, query_info)
            
            # If no good matches or low confidence, try external fallback
            if (not response.get("matches") or response["confidence"] == "low") and self.use_external:
                external_response = self.external_fallback.get_answer(question)
                
                if external_response:
                    # If we have no local matches, use only external
                    if not response.get("matches"):
                        return {
                            "answer": external_response["answer"],
                            "source": AnswerSource.EXTERNAL.value,
                            "confidence": external_response.get("confidence", "medium"),
                            "model": external_response.get("model"),
                            "query_type": query_type.value,
                            "query_understood": True
                        }
                    # Otherwise combine responses
                    response.update({
                        "external_answer": external_response["answer"],
                        "source": AnswerSource.COMBINED.value,
                        "external_model": external_response.get("model")
                    })
            
            # Add metadata
            response["available_sources"] = list(self.data_sources)
            response.update({
                "query_type": query_type.value,
                "nutritional_aspects": query_info.get('nutritional_aspects', []),
                "quantity_indicators": query_info.get('quantity_indicators', {}),
                "comparison_items": query_info.get('comparison_items', [])
            })
            
            return response
            
        except Exception as e:
            return {
                "error": f"An error occurred while processing your question: {str(e)}",
                "source": AnswerSource.LOCAL.value,
                "confidence": "low"
            } 
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ImportWarning)

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from enum import Enum
import re
from .query_processor import QueryProcessor, QueryType
from .external_fallback import ExternalFallback

# Configure numpy to use newer API
np.set_printoptions(legacy='1.13')

class AnswerSource(Enum):
    LOCAL = "local"
    EXTERNAL = "external"
    COMBINED = "combined"

class RAGEngine:
    def __init__(self, df: pd.DataFrame, use_external: bool = True):
        """Initialize the RAG engine with a dataframe of food data."""
        self.df = df
        self.use_external = use_external
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.query_processor = QueryProcessor()
        
        if self.use_external:
            self.external_fallback = ExternalFallback()
        
        # Create food contexts and embeddings
        self.food_contexts = self._create_food_contexts()
        self.embeddings = self._create_embeddings()
        self.index = self._create_index()
        
        # Track data sources
        self.data_sources = set(df['data_source'].unique()) if 'data_source' in df.columns else {'default'}
        
    def _create_food_contexts(self) -> List[str]:
        """Create context strings for each food item."""
        contexts = []
        for _, row in self.df.iterrows():
            context = f"""
            Food: {row['Voedingsmiddelnaam/Dutch food name']}
            Energy: {row['ENERCC (kcal)']} kcal/100g
            Protein: {row['PROT (g)']} g/100g
            Carbs: {row['CHO (g)']} g/100g
            Fat: {row['FAT (g)']} g/100g
            Fiber: {row.get('FIBT (g)', 'N/A')} g/100g
            Vitamin C: {row.get('VITC (mg)', 'N/A')} mg/100g
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
        
        # Define nutritional patterns
        patterns = {
            'vitamin_c': (r'vitamin\s*c', 'VITC (mg)'),
            'protein': (r'protein', 'PROT (g)'),
            'calories': (r'calorie|energy', 'ENERCC (kcal)'),
            'fiber': (r'fiber|fibre', 'FIBT (g)')
        }
        
        # Check if query matches any nutritional pattern
        for nutrient, (pattern, column) in patterns.items():
            if re.search(pattern, query_lower):
                # Sort results based on the specific nutrient value
                if column in self.df.columns:
                    sorted_results = []
                    for idx, dist, context in results:
                        try:
                            # Convert value to float and handle NaN/invalid values
                            value = self.df.iloc[idx][column]
                            value = float(value) if pd.notna(value) and str(value).replace('.', '').isdigit() else 0.0
                            # Convert distance to float and adjust based on nutrient value
                            dist_float = float(dist)
                            adjusted_dist = dist_float / (1 + value)
                            sorted_results.append((idx, adjusted_dist, context))
                        except (ValueError, TypeError):
                            # If conversion fails, use original distance
                            sorted_results.append((idx, float(dist), context))
                    return sorted(sorted_results, key=lambda x: x[1])
        
        # If no pattern matches or conversion fails, return original results with float distances
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
        
        for idx, dist, context in relevant_foods:
            food_name = self.df.iloc[idx]['Voedingsmiddelnaam/Dutch food name']
            source = self.df.iloc[idx].get('data_source', 'unknown')
            relevance = 1 / (1 + float(dist))  # Convert distance to similarity score
            
            # Only include foods with relevance above threshold
            if relevance >= min_relevance_threshold:
                foods_info.append({
                    "food": food_name,
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
        if query_type == QueryType.COMPARISON and len(query_info.get('comparison_items', [])) == 2:
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
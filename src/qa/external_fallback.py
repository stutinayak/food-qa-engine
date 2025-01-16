from typing import Optional, Dict, Any
import os
import requests

class ExternalFallback:
    """Get your Jina AI API key for free: https://jina.ai/?sui=apikey"""
    
    def __init__(self):
        self.api_key = os.getenv("JINA_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
            
    def get_answer(self, question: str) -> Optional[Dict[str, Any]]:
        """Get answer using Jina AI embeddings for complex questions"""
        if not self.api_key:
            print("Warning: JINA_API_KEY not found in environment variables")
            return None

        try:
            # Use the classifier to determine the type of question
            classify_response = requests.post(
                "https://api.jina.ai/v1/classify",
                headers=self.headers,
                json={
                    "model": "jina-embeddings-v3",
                    "input": [question],
                    "labels": [
                        "nutrition facts and ingredients",
                        "cooking techniques and recipes",
                        "food science and chemistry",
                        "dietary advice and health",
                        "non-food related question"
                    ]
                }
            )
            classify_response.raise_for_status()
            
            # Get the classification result
            classification = classify_response.json()
            if "data" in classification and classification["data"]:
                predictions = classification["data"][0]["predictions"]
                
                # Find the highest confidence prediction
                best_prediction = max(predictions, key=lambda x: x["score"])
                category = best_prediction["label"]
                confidence = best_prediction["score"]
                
                # If it's non-food related or confidence is too low
                if category == "non-food related question" or confidence < 0.3:
                    return {
                        "answer": "I apologize, but I can only answer questions related to food, nutrition, cooking, and dietary advice. Your question appears to be outside my area of expertise.",
                        "source": "jina-ai",
                        "confidence": "high" if confidence > 0.5 else "low",
                        "model": "jina-embeddings-v3"
                    }
                
                # Format response for food-related questions
                response = {
                    "answer": f"This is a question about {category.lower()}. " +
                             f"Based on the classification confidence of {confidence:.2f}, " +
                             f"I can provide information from our nutrition and food science database.",
                    "source": "jina-ai",
                    "confidence": "high" if confidence > 0.5 else "medium" if confidence > 0.3 else "low",
                    "model": "jina-embeddings-v3"
                }
                return response
                
            return None
            
        except Exception as e:
            print(f"Jina AI API error: {str(e)}")
            return None
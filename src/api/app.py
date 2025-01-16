from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import logging

from src.data.loader import DataLoader
from src.qa.rag_engine import RAGEngine, AnswerSource

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize data loader and QA engine
try:
    data_loader = DataLoader()
    df = data_loader.load_data()
    if df is None or df.empty:
        logger.error("Failed to load data: DataFrame is empty or None")
        raise ValueError("Failed to load data")
    qa_engine = RAGEngine(df=df, use_external=True)
    logger.info("Successfully initialized QA engine")
except Exception as e:
    logger.error(f"Error initializing QA engine: {str(e)}")
    raise

class Question(BaseModel):
    text: str
    type: Optional[str] = None

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/ask")
def ask_question(question: Question) -> Dict[str, Any]:
    """Answer a question about food and nutrition."""
    try:
        if not question.text:
            raise HTTPException(status_code=422, detail="Question text cannot be empty")
            
        logger.debug(f"Processing question: {question.text}")
        
        try:
            response = qa_engine.answer_question(question.text)
        except Exception as e:
            logger.error(f"QA engine error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
        
        if "error" in response:
            logger.error(f"Error in QA engine response: {response['error']}")
            raise HTTPException(status_code=404, detail=response["error"])
            
        logger.debug(f"Generated response: {response}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") 
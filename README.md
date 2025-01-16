# Food Dataset Q&A System

A sophisticated question-answering system for nutritional and food-related queries, combining local knowledge base with Jina AI capabilities.

## Quick Start

1. **Clone and Setup Environment**
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Set Up API Key**
```bash
# Get your API key from https://jina.ai/?sui=apikey
export JINA_API_KEY=your-jina-api-key
```

3. **Run the System**
```bash
# Start the FastAPI server
uvicorn src.api.app:app --reload

# Server will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

4. **Test the System**
```bash
# In a new terminal, try a simple query
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"text": "How much protein is in chicken?"}'

# Run the test suite
pytest tests/test_api_updated.py -v
```

## System Architecture

### Core Components

1. **RAG Engine** (`src/qa/rag_engine.py`)
   - Main QA processing pipeline
   - Semantic search using FAISS
   - Confidence-based answer generation
   - Integration with Jina AI

2. **Query Processor** (`src/qa/query_processor.py`)
   - Question classification
   - Information extraction
   - Pattern matching for query understanding

3. **External API Integration** (`src/qa/external_fallback.py`)
   - Jina AI integration for complex queries
   - Classification of non-food questions
   - Fallback mechanism for uncertain answers

### Data Flow

1. User submits a question
2. Query processor classifies and extracts information
3. RAG engine searches local database
4. System either:
   - Returns high-confidence local answer
   - Falls back to Jina AI for complex queries

## Detailed Setup

### Prerequisites
- Python 3.8 or higher
- Virtual environment tool
- curl (for testing API calls) or any API client

### Installation Steps

1. **Environment Setup**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Configuration**
```bash
# Set environment variables
export JINA_API_KEY=your-jina-api-key  # Get from https://jina.ai/?sui=apikey

# Optional: Add to your shell profile (.bashrc, .zshrc, etc.)
echo 'export JINA_API_KEY=your-jina-api-key' >> ~/.bashrc
```

3. **Running the Server**
```bash
# Development mode with auto-reload
uvicorn src.api.app:app --reload

# Production mode
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Verification
1. Open http://localhost:8000/docs in your browser
2. Try the health check endpoint
3. Test with a sample question

## Usage

### API Endpoints

1. **Question Answering**
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"text": "How much protein is in chicken?"}'
```

2. **Health Check**
```bash
curl "http://localhost:8000/health"
```

### Example Questions

1. Nutritional Queries:
   - "How much protein is in chicken?"
   - "What is the calorie content of bananas?"

2. Comparisons:
   - "Compare the protein content of chicken and fish"
   - "Which has more vitamin C, oranges or bananas?"

3. Recommendations:
   - "What are some high-protein foods?"
   - "List foods low in calories"

### Response Format

```json
{
    "answer": "Detailed answer text",
    "confidence": "high/medium/low",
    "source": "local/jina-ai",
    "matches": [
        {
            "food": "Food name",
            "relevance": 0.95,
            "context": "Nutritional information"
        }
    ],
    "query_type": "Query classification",
    "nutritional_aspects": ["Identified nutrients"]
}
```

## File Structure

```
src/
├── api/
│   └── app.py           # FastAPI application
├── data/
│   └── loader.py        # Data loading utilities
└── qa/
    ├── __init__.py
    ├── rag_engine.py    # Main QA engine
    ├── query_processor.py # Query understanding
    └── external_fallback.py # Jina AI integration
```

## Performance Considerations

1. **Local Processing**
   - FAISS index for fast similarity search
   - Cached embeddings for quick retrieval
   - Pattern-based query classification

2. **External API Usage**
   - Rate limiting for Jina AI calls
   - Fallback mechanism for API failures
   - Confidence thresholds for external calls

## Testing

The system uses pytest for testing:
```bash
# Run all tests
pytest tests/test_api_updated.py -v

# Run specific test
pytest tests/test_api_updated.py -v -k "test_health_check"
```

Key test files:
- `tests/test_api_updated.py`: API endpoint tests
- `tests/conftest.py`: Test configurations

## Troubleshooting

1. **Server Already Running**
   ```bash
   # If you see "Address already in use"
   lsof -i :8000  # Find process using port 8000
   kill -9 <PID>  # Kill the process
   ```

2. **API Key Issues**
   - Verify key is set: `echo $JINA_API_KEY`
   - Try re-exporting the key
   - Check Jina AI dashboard for key status

3. **Dependencies Issues**
   ```bash
   # Reinstall dependencies
   pip uninstall -r requirements.txt -y
   pip install -r requirements.txt
   ```
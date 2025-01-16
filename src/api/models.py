from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class QuestionType(str, Enum):
    ENERGY = "energy"
    TRACE = "trace"
    VITAMIN_C = "vitamin_c"
    PROTEIN = "protein"
    FIBER = "fiber"

class Question(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        max_length=200,
        description="The question about food data"
    )
    type: Optional[QuestionType] = Field(
        None,
        description="Type of question (optional)"
    )

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0" 
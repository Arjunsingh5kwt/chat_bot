from pydantic import BaseModel
from typing import Optional
from typing import List

class SimilarityResult(BaseModel):
    sentence: str
    
    
# Define the model for input query
class Query(BaseModel):
    query: str

class UserResponse(BaseModel):
    response: List[SimilarityResult]
    
# Define the model for input query
class Update_bot(BaseModel):
    title: Optional[str] = None
    slugs: Optional[str] = None

class Update_bot_response(BaseModel):
    response: Optional[str] = None
    
    
    
  
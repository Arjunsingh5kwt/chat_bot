from pydantic import BaseModel
from typing import Optional
from typing import List
from typing import List, Union, Optional

class SimilarityResult(BaseModel):
    id: int
    answer: str
    slugs: str
    types: str
    

    
# Define the model for input query
class Query(BaseModel):
    query: str

class UserResponse(BaseModel):
    response: Union[List[SimilarityResult], str]
    
# Define the model for input query
class Update_bot(BaseModel):
    id: Optional[int] = None
    title: Optional[str] = None
    answer: Optional[str] = None
    slugs: Optional[str] = None
    types: Optional[str] = None

class Update_bot_response(BaseModel):
    response: Optional[str] = None
    
    
    
  
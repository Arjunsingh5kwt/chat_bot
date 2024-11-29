from pydantic import BaseModel
from typing import Optional


# Define the model for input query
class Query(BaseModel):
    query: str

class UserResponse(BaseModel):
    response:  Optional[str] = None
    
# Define the model for input query
class Update_bot(BaseModel):
    title: Optional[str] = None
    slug: Optional[str] = None

class Update_bot_response(BaseModel):
    response: Optional[str] = None
    
    
    
  
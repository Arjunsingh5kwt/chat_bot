from fastapi import FastAPI, Body
from pydantic import BaseModel
from src import models  # Import the model for query processing
import uvicorn

app = FastAPI()  
    
# Define the model for input query
class Query(BaseModel):
    query: str
class userresponse(BaseModel):
    response: str
    
    
# Load the model and dataset
sentence_model = models.load_model()
file_path = r"cleaned_file3.csv"
data = models.load_csv_data(file_path)
sentences = list(data["question"])

@app.post("/hello",response_model=userresponse)  # Route to handle query processing
async def get_answer(request_data: Query = Body(...)):
    query = request_data.query
    if query:
        most_similar_sentence, most_similar_idx, similarity_score = models.find_most_similar(query, sentences, sentence_model)

        # Apply threshold
        threshold = 0.50
        if similarity_score >= threshold:
            answer = data.iloc[most_similar_idx]["answer"]
            return {"response": answer}
        else:
            return {"response": "Sorry, I don't understand the question."}
    else:
        return {"response": "Query cannot be empty."}
    
    
    json_resp = {
        
        "response":response
    }
    
    # return json_resp
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
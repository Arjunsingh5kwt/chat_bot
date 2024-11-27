from fastapi import FastAPI, Body
from pydantic import BaseModel
from src import models  # Import the model for query processing
import uvicorn
import pandas as pd

app = FastAPI()

# Define the model for input query
class Query(BaseModel):
    query: str

class UserResponse(BaseModel):
    response: str
    content: str

# Load the model and dataset
sentence_model = models.load_model()

# Paths for CSV and pickle
file_path = r"updated_extracted_data.csv"
embedding_pickle_path = r"sentence_embeddings.pkl"

# Load data and precomputed embeddings
data = models.load_csv_data(file_path)
sentences = list(data["question"])
embeddings = models.encode_sentences(sentence_model, sentences, embedding_pickle_path)

@app.post("/chatbot", response_model=UserResponse)  # Route to handle query processing
async def get_answer(request_data: Query = Body(...)):
    query = request_data.query
    if query:
        most_similar_sentence, most_similar_idx, similarity_score = models.find_most_similar(
            query, sentences, sentence_model, embeddings
        )

        # Apply threshold
        threshold = 0.50
        if similarity_score >= threshold:
            answer = data.iloc[most_similar_idx]["answer"]
            content = data.iloc[most_similar_idx]["content"]
             # Check if the content cell is empty
            # Check if content or answer is empty or invalid
            if pd.isna(content) or content is None or content.strip() == "":
                content = "N/A"
            if pd.isna(answer) or answer is None or answer.strip() == "":
                answer = "N/A"
            return {"response": answer, "content": content}
        else:
            return {"response": "Sorry, I don't understand the question.", "content": "N/A"}
    else:
        return {"response": "Query cannot be empty.", "content": "N/A"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

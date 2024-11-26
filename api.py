from fastapi import FastAPI, Body
from pydantic import BaseModel
from src import models  # Import the model for query processing
import uvicorn

app = FastAPI()

# Define the model for input query
class Query(BaseModel):
    query: str

class UserResponse(BaseModel):
    response: str

# Load the model and dataset
sentence_model = models.load_model()

# Paths for CSV and pickle
file_path = r"extracted_title_slug.csv"
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
            return {"response": answer}
        else:
            return {"response": "Sorry, I don't understand the question."}
    else:
        return {"response": "Query cannot be empty."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

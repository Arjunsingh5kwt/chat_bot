from fastapi import FastAPI, Body
from pydantic import BaseModel
from src import models  # Import the model for query processing
import uvicorn
import pandas as pd
import pickle
from src.endpoints.endpoints import (
    Query,
    UserResponse,
    Update_bot,
    Update_bot_response)

app = FastAPI()

# Load the model and dataset
sentence_model = models.load_model()
file_path = "updated_extracted_data.csv"
embedding_pickle_path = "sentence_embeddings.pkl"


# Load data and precomputed embeddings

@app.post("/chatbot", response_model=UserResponse)  # Route to handle query processing
async def get_answer(request_data: Query = Body(...)):
    query = request_data.query
    # Paths for CSV and pickle
    data = models.load_csv_data(file_path)
    sentences = list(data["title"])
    embeddings = models.encode_sentences(sentence_model, sentences, embedding_pickle_path)
    if query:
        most_similar_sentence, most_similar_idx, similarity_score = models.find_most_similar(
            query, sentences, sentence_model, embeddings
        )

        # Apply threshold
        threshold = 0.50
        if similarity_score >= threshold:
            slugs = data.iloc[most_similar_idx]["slugs"]
             # Check if the content cell is empty
            # Check if content or answer is empty or invalid
            if pd.isna(slugs) or slugs is None or slugs.strip() == "":
                answer = "N/A"
            return {"response": slugs}
        else:
            return {"response": "Sorry, I don't understand the question."}
    else:
        return {"response": "Query cannot be empty."}



@app.post("/update-blog", response_model=Update_bot_response)
async def update_blog(
    request_data: Update_bot = Body(...)):
    title = request_data.title
    slugs = request_data.slugs
    
    # Read existing CSV file
    df = pd.read_csv(file_path,encoding='latin')
    
    
    # Add a new post
    new_row = {"title": " ".join(title.split()),  # Remove extra spaces, keep single spaces
           "slugs": " ".join(slugs.split())}
    # Check if the title already exists
    if title.strip().lower() in df["title"].str.strip().str.lower().values:
        # Fetch the existing row(s) with the same title
        existing_row = df[df["title"] == title] 
        if not existing_row.empty: 
            # Check if the slugs and content are different
            if not (existing_row["slugs"].str.strip().str.lower().values[0] == slugs.strip().lower()): 
                # Update the slugs and content for the existing title
                df.loc[df["title"] == title, "slugs"] = " ".join(slugs.split())
                print(f"Updated the existing entry for title: {title}")
            else:
                print(f"No changes needed for title: {title}, slugs is same.")
        else:
            print("No changes needed for title: {title}, slugs is same..")
    else:
        # Append the new row if title doesn't exist
        df = df._append(new_row, ignore_index=True)
        print(f"Appended new entry with title: {title}")

    # Save the updated dataframe back to the CSV
    df.to_csv(file_path, index=False, encoding='utf-8')
    
    # Generate embeddings for the content and update the pickle files
    # PICKLE_FILE = "sentence_embeddings.pkl"  # This defines the file name

    sentences = df['title'].tolist()
    embeddings =sentence_model.encode(sentences)
    with open(embedding_pickle_path, "wb") as f:
        pickle.dump(embeddings, f)
    jsonresponse= {
        "response":"done"
    }
    # response = {"slugs": slugs, "content": content}
    return jsonresponse

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
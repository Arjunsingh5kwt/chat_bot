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
        # Find the top 3 most similar sentences
        top_k_results = models.find_top_k_similar(query, sentences, sentence_model, embeddings, k=3)
        
        # Extract the top 3 matches (sentence, index, similarity score)
        top_matches = [
            {
                "id": data.iloc[result[1]]["id"],
                "answer": data.iloc[result[1]]["answer"],
                "slugs": data.iloc[result[1]]["slugs"],
                "types": data.iloc[result[1]]["types"],
                "similarity": result[2]
            }
            for result in top_k_results
        ]
        
        # Apply threshold
        threshold = 0.50
        filtered_matches = [
            match for match in top_matches if match["similarity"] >= threshold
        ]
        
        if filtered_matches:
            # Limit the answer to 30-35 words
            for match in filtered_matches:
                # Truncate the answer to 30-35 words
                answer_words = match["answer"].split()
                if len(answer_words) > 35:
                    match["answer"] = " ".join(answer_words[:35]) + "..."
                
                # Include a fallback if slugs are empty or invalid
                if pd.isna(match["slugs"]) or match["slugs"].strip() == "":
                    match["slugs"] = "N/A"
                # Include a fallback if types are empty or invalid
                if pd.isna(match["types"]) or match["types"].strip() == "":
                    match["types"] = "N/A"
            
            return {"response": filtered_matches}
        else:
            return {"response": "Sorry, I don't understand the question."}
    else:
        return {"response": "Query cannot be empty."}



 # Read existing CSV file
df = pd.read_csv(file_path, encoding='latin')

@app.post("/update-blog", response_model=Update_bot_response)
async def update_blog(
    request_data: Update_bot = Body(...)):
    id = request_data.id
    title = request_data.title
    answer = request_data.answer
    slugs = request_data.slugs
    types = request_data.types

   
    

    # Add a new post
    new_row = {
        "id": id,  # Remove extra spaces, keep single spaces
        "title": " ".join(title.split()),  # Remove extra spaces, keep single spaces
        "answer": " ".join(answer.split()),
        "types": " ".join(types.split()),
        "slugs": " ".join(slugs.split())
    }

    # Check if the title already exists
    if title.strip().lower() in df["title"].str.strip().str.lower().values:
        # Fetch the existing row(s) with the same title
        existing_row = df[df["title"].str.strip().str.lower() == title.strip().lower()]
        
        if not existing_row.empty: 
            #for the blogs
            if types.strip().lower() == "blogs":
                # Check if the id already exists in the "blogs" type
                if id in df[df["types"].str.strip().str.lower() == "blogs"]["id"].values:
                    # Check if the title is also the same for this id
                    existing_title = df[(df["types"].str.strip().str.lower() == "blogs") & (df["id"] == id)]["title"]
                    if not existing_title.empty and existing_title.values[0].strip().lower() != title.strip().lower():
                        return {"response": "Error: ID already exists for a different title in 'blogs'."}
                    
                # Check if the slugs and content are different
                if not (existing_row["answer"].str.strip().str.lower().values[0] == answer.strip().lower()) or \
                not (existing_row["slugs"].str.strip().str.lower().values[0] == slugs.strip().lower()) or \
                not (existing_row["types"].str.strip().str.lower().values[0] == types.strip().lower()): 
                    
                    # If answer, slugs, or types are empty, set them to a single space " "
                    answer = " " if not answer.strip() else answer
                    slugs = " " if not slugs.strip() else slugs
                    types = " " if not types.strip() else types
                    
                    # Update the slugs and content for the existing title
                    df.loc[df["title"].str.strip().str.lower() == title.strip().lower(), "id"] = id
                    df.loc[df["title"].str.strip().str.lower() == title.strip().lower(), "answer"] = " ".join(answer.split())
                    df.loc[df["title"].str.strip().str.lower() == title.strip().lower(), "slugs"] = " ".join(slugs.split())
                    df.loc[df["title"].str.strip().str.lower() == title.strip().lower(), "types"] = " ".join(types.split())
                    print(f"Updated the existing entry for title: {title}")
                else:
                    print(f"No changes needed for title: {title}, answer and slugs are the same.")

            #for faq  
            if types.strip().lower() == "faq":
                if id in df[df["types"].str.strip().str.lower() == "faq"]["id"].values:
                    # Check if the title is also the same for this id
                    existing_title = df[(df["types"].str.strip().str.lower() == "faq") & (df["id"] == id)]["title"]
                    if not existing_title.empty and existing_title.values[0].strip().lower() != title.strip().lower():
                        return {"response": "Error: ID already exists for a different title in 'faq'."}
                    
                # Check if the slugs and content are different
                if not (existing_row["answer"].str.strip().str.lower().values[0] == answer.strip().lower()) or not (existing_row["slugs"].str.strip().str.lower().values[0] == slugs.strip().lower()) or not (existing_row["types"].str.strip().str.lower().values[0] == types.strip().lower()): 
                
                    # If answer, slugs, or types are empty, set them to a single space " "
                    answer = " " if not answer.strip() else answer
                    slugs = " " if not slugs.strip() else slugs
                    types = " " if not types.strip() else types
                    
                    # Update the slugs and content for the existing title
                    df.loc[df["title"].str.strip().str.lower() == title.strip().lower(), "id"] = id
                    df.loc[df["title"].str.strip().str.lower() == title.strip().lower(), "answer"] = " ".join(answer.split())
                    df.loc[df["title"].str.strip().str.lower() == title.strip().lower(), "slugs"] = " ".join(slugs.split())
                    df.loc[df["title"].str.strip().str.lower() == title.strip().lower(), "types"] = " ".join(types.split())
                    print(f"Updated the existing entry for title: {title}")
                else:
                    print(f"No changes needed for title: {title}, answer and slugs are the same.")
        else:
            print(f"No changes needed for title: {title}, it could be missing data.")
    else:
        # Append the new row if title doesn't exist
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"Appended new entry with title: {title}")

    # Save the updated dataframe back to the CSV
    df.to_csv(file_path, index=False, encoding='utf-8')

    # Generate embeddings for the content and update the pickle files
    sentences = df['title'].tolist()
    embeddings = sentence_model.encode(sentences)
    with open(embedding_pickle_path, "wb") as f:
        pickle.dump(embeddings, f)

    jsonresponse = {
        "response": "done"
    }

    return jsonresponse


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
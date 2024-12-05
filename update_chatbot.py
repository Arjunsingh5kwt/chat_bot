# from fastapi import FastAPI, Body
# import os  
# from src import models  # Import the model for query processing
# import uvicorn
# import pandas as pd
# from pydantic import BaseModel
# from typing import Optional
# import pickle

# app = FastAPI()

# sentence_model = models.load_model()

# # Paths for the CSV and pickle files
# CSV_FILE = "extracted_title_slug.csv"
# PICKLE_FILE = "sentence_embeddings.pkl"


# # Ensure the CSV file exists
# if not os.path.exists(CSV_FILE):
#     pd.DataFrame(columns=["title", "slug", "content"]).to_csv(CSV_FILE, index=False)

# @app.post("/update-blog", response_model=Update_bot_response)
# async def update_blog(
#     request_data: Update_bot = Body(...)):
#     title = request_data.query
#     slug = request_data.answer
#     content = request_data.content
    
#     # Read existing CSV file
#     df = pd.read_csv(CSV_FILE,encoding='latin')
    
    
#     # Add a new post
#     new_row = {"title": title, "slug": slug, "content": content}
#     df = df._append(new_row, ignore_index=True)
    
#     # Save the updated dataframe back to the CSV
#     df.to_csv(CSV_FILE, index=False)
    
#     # Generate embeddings for the content and update the pickle file
#     PICKLE_FILE = "sentence_embeddings.pkl"  # This defines the file name

#     sentences = df['title'].tolist()
#     embeddings =sentence_model.encode(sentences)
#     with open(PICKLE_FILE, "wb") as f:
#         pickle.dump(embeddings, f)
#     jsonresponse= {
#         "response":"done"
#     }
#     # response = {"slug": slug, "content": content}
#     return jsonresponse

# # Run the FastAPI application with uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=5000)
import pickle
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

# Save an object using pickle
def save_with_pickle(obj, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

# Load an object using pickle
def load_with_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Load SentenceTransformer model
def load_model():
    return SentenceTransformer('all-MiniLM-L12-v2')

# Encode sentences and optionally save embeddings
def encode_sentences(model, sentences, pickle_path=None):
    if pickle_path:
        try:
            # Attempt to load embeddings from pickle
            return load_with_pickle(pickle_path)
        except FileNotFoundError:
            print(f"Pickle file {pickle_path} not found. Generating embeddings...")
            embeddings = model.encode(sentences, convert_to_tensor=True)
            save_with_pickle(embeddings, pickle_path)
            return embeddings
    else:
        return model.encode(sentences, convert_to_tensor=True)

def find_most_similar(query, sentences, model, embeddings):
    if not query.strip() or not sentences:
        raise ValueError("Query or sentences list cannot be empty.")
    
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_similarities = util.pytorch_cos_sim(query_embedding, embeddings).cpu().numpy().flatten()
    most_similar_idx = np.argmax(cosine_similarities)
    
    return sentences[most_similar_idx], most_similar_idx, cosine_similarities[most_similar_idx]

def load_csv_data(file_path):
    """
    Load a CSV file, handling potential encoding issues.
    :param file_path: Path to the CSV file
    :return: DataFrame with the loaded data
    """
    try:
        return pd.read_csv(file_path, encoding='latin-1')
    except UnicodeDecodeError:
        # Try with a different encoding
        print("Failed to decode the file with latin-1 . Retrying with 'utf-8' encoding...")
        return pd.read_csv(file_path, encoding='utf-8')

# Initialize BM25 model
def initialize_bm25(sentences):
    tokenized_sentences = [sentence.split() for sentence in sentences]
    return BM25Okapi(tokenized_sentences)

# Compute BM25 scores
def bm25_scores(bm25_model, query):
    tokenized_query = query.split()
    return bm25_model.get_scores(tokenized_query)

# Initialize TF-IDF vectorizer
def initialize_tfidf(sentences):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return vectorizer, tfidf_matrix

# Compute TF-IDF cosine similarity scores
def tfidf_cosine_scores(tfidf_vectorizer, tfidf_matrix, query):
    query_vector = tfidf_vectorizer.transform([query])
    return (tfidf_matrix @ query_vector.T).toarray().flatten()

# Compute hybrid similarity
def hybrid_similarity(query, sentences, bm25_model, tfidf_vectorizer, tfidf_matrix, alpha=0.5):
    """
    Compute a hybrid similarity score using BM25 and cosine similarity.

    :param query: The query string
    :param sentences: List of predefined sentences
    :param bm25_model: Initialized BM25 model
    :param tfidf_vectorizer: Initialized TF-IDF vectorizer
    :param tfidf_matrix: Precomputed TF-IDF matrix
    :param alpha: Weight for cosine similarity (1-alpha is for BM25)
    :return: Best matching sentence, index, and hybrid similarity score
    """
    # Compute BM25 scores
    bm25_scores_list = bm25_scores(bm25_model, query)

    # Compute cosine similarity scores
    cosine_scores = tfidf_cosine_scores(tfidf_vectorizer, tfidf_matrix, query)

    # Normalize scores
    bm25_scores_list = (bm25_scores_list - min(bm25_scores_list)) / (max(bm25_scores_list) - min(bm25_scores_list))
    cosine_scores = (cosine_scores - min(cosine_scores)) / (max(cosine_scores) - min(cosine_scores))

    # Combine scores
    hybrid_scores = alpha * cosine_scores + (1 - alpha) * bm25_scores_list

    # Find the best match
    best_idx = np.argmax(hybrid_scores)
    return sentences[best_idx], best_idx, hybrid_scores[best_idx]

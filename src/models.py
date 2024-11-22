from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

def load_model():
    return SentenceTransformer('all-MiniLM-L12-v2')

def encode_sentences(model, sentences):
    return model.encode(sentences, convert_to_tensor=True)

def find_most_similar(query, sentences, model):
    if not query.strip() or not sentences:
        raise ValueError("Query or sentences list cannot be empty.")
    
    query_embedding = model.encode(query, convert_to_tensor=True)
    sentence_embeddings = encode_sentences(model, sentences)
    cosine_similarities = util.pytorch_cos_sim(query_embedding, sentence_embeddings).cpu().numpy().flatten()
    most_similar_idx = np.argmax(cosine_similarities)
    
    return sentences[most_similar_idx], most_similar_idx, cosine_similarities[most_similar_idx]

def load_csv_data(file_path):
    return pd.read_csv(file_path)

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
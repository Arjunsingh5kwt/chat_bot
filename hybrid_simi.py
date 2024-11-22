from src import models
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np

# App title
print("Customer Support Chatbot")

# Load the model and data
sentence_model = models.load_model()
file_path = r"C:\Users\Admin\OneDrive\Desktop\chatbot_using_python\Customer_Support_Questions_and_Answers.csv"
data = models.load_csv_data(file_path)

# Load predefined sentences
if "Question" in data.columns:
    sentences = list(data["Question"])
else:
    raise ValueError("The dataset does not contain a 'Question' column.")

# Precompute TF-IDF and BM25 representations
print("Initializing text retrieval models...")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)  # TF-IDF matrix for cosine similarity

tokenized_sentences = [sentence.split() for sentence in sentences]
bm25_model = BM25Okapi(tokenized_sentences)  # BM25 model initialization


def hybrid_similarity(query, alpha=0.5):
    """
    Compute a hybrid similarity score using BM25 and cosine similarity.

    :param query: The query string
    :param alpha: Weight for cosine similarity (1-alpha is for BM25)
    :return: The best-matching sentence, its index, and the combined score
    """
    # Compute BM25 scores
    bm25_scores = bm25_model.get_scores(query.split())

    # Compute cosine similarity scores using TF-IDF
    query_tfidf = tfidf_vectorizer.transform([query])
    cosine_scores = (tfidf_matrix @ query_tfidf.T).toarray().flatten()

    # Normalize scores for combination
    bm25_scores = (bm25_scores - min(bm25_scores)) / (max(bm25_scores) - min(bm25_scores))
    cosine_scores = (cosine_scores - min(cosine_scores)) / (max(cosine_scores) - min(cosine_scores))

    # Combine scores
    hybrid_scores = alpha * cosine_scores + (1 - alpha) * bm25_scores

    # Find the best match
    best_idx = hybrid_scores.argmax()
    return sentences[best_idx], best_idx, hybrid_scores[best_idx]

while True:
    query = input("Enter your question (or type 'exit' to quit): ").strip()
    if query.lower() == "exit":
        print("Exiting the chatbot. Goodbye!")
        break

    if query:
        try:
            # Find the most similar question using hybrid similarity
            alpha = 0.5  # Adjust the weight for cosine similarity vs BM25
            best_sentence, best_idx, hybrid_score = hybrid_similarity(query, alpha)

            # Print the most similar question and hybrid score
            print(f"Most similar question: {best_sentence}")
            print(f"Hybrid similarity score: {hybrid_score:.2f}")

            # Apply threshold
            threshold = 0.50
            if hybrid_score >= threshold:
                if "Answer" in data.columns:
                    answer = data.iloc[best_idx]["Answer"]
                    print(f"Answer: {answer}")
                else:
                    print("The dataset does not contain an 'Answer' column.")
            else:
                print("Sorry, I don't understand the question.")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("Please enter a valid question.")

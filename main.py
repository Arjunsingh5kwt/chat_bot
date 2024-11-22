import streamlit as st
from src import models

# App title
st.title("Customer Support Chatbot")

# Load the model and data
sentence_model = models.load_model()
file_path = r"C:\Users\Admin\OneDrive\Desktop\chatbot_using_python\Customer_Support_Questions_and_Answers.csv"
data = models.load_csv_data(file_path)

# Load predefined sentences
if "Question" in data.columns:
    sentences = list(data["Question"])
else:
    st.error("The dataset does not contain a 'Question' column.")
    st.stop()

# Input query from the user
query = st.text_input("Enter your question:")
submit = st.button("Submit")

# Process the query
if submit:
    if query.strip():
        most_similar_sentence, most_similar_idx, similarity_score = models.find_most_similar(query, sentences, sentence_model)

        # Commenting out similarity score and most similar question
        st.write(f"Most similar question: {most_similar_sentence}")
        st.write(f"Similarity score: {similarity_score:.2f}")

        # Apply threshold
        threshold = 0.70
        if similarity_score >= threshold:
            if "Answer" in data.columns:
                answer = data.iloc[most_similar_idx]["Answer"]
                st.write(f"Answer: {answer}")
            else:
                st.write("The dataset does not contain an 'Answer' column.")
        else:
            st.write("Sorry, I don't understand the question.")
    else:
        st.error("Please enter a valid question.")

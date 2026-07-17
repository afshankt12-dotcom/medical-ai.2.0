import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    MEDQUAD_PATH,
    CHATBOT_DATASET_PATH,
    MEDQUAD_HF_PATH,
    PUBMEDQA_PATH,
)


class Retriever:

    def __init__(self):

        self.questions = []
        self.answers = []

        self.load_datasets()

        print("Building TF-IDF index...")

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.question_vectors = self.vectorizer.fit_transform(
            self.questions
        )

        print("Retriever Ready!")

    def load_datasets(self):

        # Unified Medical Chatbot Dataset
        chatbot = pd.read_csv(CHATBOT_DATASET_PATH)

        self.questions.extend(
            chatbot["question"].fillna("").astype(str).tolist()
        )

        self.answers.extend(
            chatbot["answer"].fillna("").astype(str).tolist()
        )

        print(f"Loaded {len(self.questions)} questions from merged dataset.")

    def search(self, user_question):

        query_vector = self.vectorizer.transform(
            [user_question]
        )

        scores = cosine_similarity(
            query_vector,
            self.question_vectors
        )[0]

        best_index = scores.argmax()

        confidence = float(scores[best_index])

        answer = self.answers[best_index]

        return answer, round(confidence, 4)
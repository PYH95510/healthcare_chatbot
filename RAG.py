import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle


class RAGRetriever:
    def __init__(self, qa_path, quiz_path, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize RAG Retriever by loading datasets and setting up FAISS."""
        self.qa_path = qa_path
        self.quiz_path = quiz_path
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.df_combined = None

    def load_data(self):
        """Load and merge Q&A and Quiz datasets for RAG processing."""
        df_kaggle = pd.read_csv(self.qa_path).dropna(subset=["short_question", "short_answer"])
        df_quiz = pd.read_csv(self.quiz_path).dropna(subset=["question", "exp"])

        # Combine datasets for FAISS indexing
        self.df_combined = pd.DataFrame({
            "text": df_kaggle["short_question"].tolist() + df_quiz["question"].tolist(),
            "context": df_kaggle["short_answer"].tolist() + df_quiz["exp"].tolist()
        })
        print("Datasets loaded and combined.")

    def build_faiss_index(self):
        """Create FAISS index with embedded questions and expert explanations."""
        if self.df_combined is None:
            raise ValueError("Data not loaded. Run load_data() first.")

        text_embeddings = self.model.encode(self.df_combined["text"].tolist(), convert_to_numpy=True)

        dimension = text_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(text_embeddings)

        # Save FAISS index and metadata
        faiss.write_index(self.index, "faiss_index_combined.bin")
        with open("qa_quiz_metadata.pkl", "wb") as f:
            pickle.dump(self.df_combined, f)

        print("FAISS index built and saved.")

    def load_faiss_index(self):
        """Load FAISS index and dataset metadata."""
        self.index = faiss.read_index("faiss_index_combined.bin")
        with open("qa_quiz_metadata.pkl", "rb") as f:
            self.df_combined = pickle.load(f)
        print("FAISS index loaded.")

    def retrieve_context(self, user_query, top_k=3):
        """Retrieve relevant context for a given user query."""
        if self.index is None or self.df_combined is None:
            raise ValueError("FAISS index not loaded. Run load_faiss_index() first.")

        query_embedding = self.model.encode([user_query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)

        results = [self.df_combined.iloc[idx]["context"] for idx in indices[0] if idx < len(self.df_combined)]
        return results

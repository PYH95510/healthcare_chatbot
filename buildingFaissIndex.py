from RAG import RAGRetriever

# Step 1: Instantiate the RAG Retriever
rag = RAGRetriever("dataset/kaggle_healthcare_data.csv", "dataset/multiple_healthcare_data.csv")

# Step 2: Load data and build FAISS index (Run this only once!)
rag.load_data()
rag.build_faiss_index()  # This creates and saves the FAISS index

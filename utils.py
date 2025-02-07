import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def load_embeddings(data_path: str, embeddings_path: str):
    """Load the dataset and precomputed embeddings."""
    df = pd.read_csv(data_path)
    embeddings = np.load(embeddings_path)
    return df, embeddings

def recommend_products(query: str, df: pd.DataFrame, embeddings: np.ndarray, top_k: int = 5):
    """Find top-k product recommendations based on query similarity."""
    # Load model for query embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])

    # Compute cosine similarity
    similarity_scores = cosine_similarity(query_embedding, embeddings)[0]
    indices = np.argsort(similarity_scores)[::-1][:top_k]

    # Fetch product details
    recommendations = df.iloc[indices][["Title", "Category", "Price"]].to_dict(orient="records")
    return recommendations

import tkinter as tk
from tkinter import ttk, messagebox
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import os

# Load the dataset
df = pd.read_csv('./data/nike_data_2022_09.csv')

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the product names (run this once)
if not os.path.exists('./data/product_embeddings.npy'):
    embeddings = model.encode(df['name'].tolist(), convert_to_numpy=True)
    np.save('./data/product_embeddings.npy', embeddings)

# Load precomputed embeddings
embeddings = np.load('./data/product_embeddings.npy')

# Initialize NearestNeighbors model
nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
nn_model.fit(embeddings)

def recommend_products(query, top_k=5):
    """
    Recommend products based on a user query.

    Args:
        query (str): The user's input query.
        top_k (int): Number of recommendations to return.

    Returns:
        List[Dict]: List of recommended products with details.
    """
    query_embedding = model.encode([query])[0]
    distances, indices = nn_model.kneighbors([query_embedding], n_neighbors=top_k)
    recommendations = df.iloc[indices[0]].to_dict(orient='records')
    return recommendations

# Tkinter UI Setup
def get_recommendations():
    query = query_entry.get().strip()
    if not query:
        messagebox.showerror("Error", "Please enter a query!")
        return

    try:
        recommendations = recommend_products(query)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Top Recommendations:\n\n")
        for idx, product in enumerate(recommendations, start=1):
            result_text.insert(tk.END, f"{idx}. {product['name']}\n")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Main UI Window
root = tk.Tk()
root.title("AI Stylist")

# Styling
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 10))
style.configure("TLabel", font=("Helvetica", 12))

# Widgets
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky="NSEW")

query_label = ttk.Label(frame, text="Enter your query:")
query_label.grid(row=0, column=0, padx=5, pady=5, sticky="W")

query_entry = ttk.Entry(frame, width=40)
query_entry.grid(row=0, column=1, padx=5, pady=5, sticky="W")

search_button = ttk.Button(frame, text="Search", command=get_recommendations)
search_button.grid(row=0, column=2, padx=5, pady=5)

result_label = ttk.Label(frame, text="Recommendations:")
result_label.grid(row=1, column=0, padx=5, pady=5, sticky="NW")

result_text = tk.Text(frame, height=15, width=60, wrap="word", font=("Helvetica", 10))
result_text.grid(row=1, column=1, columnspan=2, padx=5, pady=5)

# Start the application
root.mainloop()


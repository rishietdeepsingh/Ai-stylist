import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import requests
from io import BytesIO
from PIL import Image, ImageTk

# Load the dataset
df = pd.read_csv('data/adidas_usa.csv')

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the product names and descriptions
if 'embeddings.npy' not in df.columns:
    embeddings = model.encode((df['name'] + " " + df['description']).tolist(), convert_to_numpy=True)
    np.save('embeddings.npy', embeddings)
else:
    embeddings = np.load('embeddings.npy')

# Initialize the Nearest Neighbors model
nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
nn_model.fit(embeddings)

# Function to get recommendations
def get_recommendations(query, top_k=5):
    query_embedding = model.encode([query])[0]
    distances, indices = nn_model.kneighbors([query_embedding], n_neighbors=top_k)
    recommendations = df.iloc[indices[0]]
    return recommendations

# Function to load and display images
def load_image(image_url):
    try:
        # If the image URL is a valid link
        response = requests.get(image_url)
        response.raise_for_status()  # Raise error for bad status codes (e.g., 404, 500)
        img = Image.open(BytesIO(response.content))
        img = img.resize((100, 100))  # Resize image to fit the UI
        img = ImageTk.PhotoImage(img)
        return img
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from URL {image_url}: {e}")
        return None
    except IOError as e:
        print(f"Error loading image {image_url}: {e}")
        return None

# Create the Tkinter UI
def recommend():
    query = query_entry.get()
    if not query.strip():
        messagebox.showerror("Input Error", "Please enter a search query!")
        return
    
    # Get recommendations
    recommendations = get_recommendations(query)
    
    # Clear previous results
    for widget in result_frame.winfo_children():
        widget.destroy()
    
    # Display recommendations
    for idx, row in recommendations.iterrows():
        name_label = tk.Label(result_frame, text=f"Product: {row['name']}", font=("Arial", 12, "bold"))
        description_label = tk.Label(result_frame, text=f"Description: {row['description']}", font=("Arial", 10))
        price_label = tk.Label(result_frame, text=f"Price: {row['selling_price']} {row['currency']}", font=("Arial", 10, "italic"))
        
        # Display image
        img_url = row['images']
        img = load_image(img_url)
        if img:
            image_label = tk.Label(result_frame, image=img)
            image_label.image = img  # Keep a reference to the image
            image_label.pack(anchor="w", pady=2)

        name_label.pack(anchor="w", pady=2)
        description_label.pack(anchor="w", pady=2)
        price_label.pack(anchor="w", pady=2)

        separator = ttk.Separator(result_frame, orient="horizontal")
        separator.pack(fill="x", pady=5)

# Initialize the main application window
app = tk.Tk()
app.title("AI Stylist - Product Recommendations")
app.geometry("600x600")

# Query input
query_label = tk.Label(app, text="Enter search query:", font=("Arial", 14))
query_label.pack(pady=10)

query_entry = tk.Entry(app, font=("Arial", 14), width=50)
query_entry.pack(pady=5)

recommend_button = tk.Button(app, text="Get Recommendations", font=("Arial", 14), command=recommend)
recommend_button.pack(pady=10)

# Results section
result_frame = tk.Frame(app)
result_frame.pack(fill="both", expand=True, pady=10, padx=10)

scrollbar = tk.Scrollbar(result_frame, orient="vertical")
scrollbar.pack(side="right", fill="y")

result_canvas = tk.Canvas(result_frame, yscrollcommand=scrollbar.set)
result_canvas.pack(side="left", fill="both", expand=True)

scrollbar.config(command=result_canvas.yview)

result_content = tk.Frame(result_canvas)
result_canvas.create_window((0, 0), window=result_content, anchor="nw")

result_content.bind("<Configure>", lambda e: result_canvas.configure(scrollregion=result_canvas.bbox("all")))

# Run the application
app.mainloop()
